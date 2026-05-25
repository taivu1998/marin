# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
import time
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
from levanter.models.lm_model import LmHeadModel
from marin.rl.environments.inference_ctx.vllm import InferenceMode, vLLMInferenceContext, vLLMInferenceContextConfig

logger = logging.getLogger(__name__)


# Allow vLLM to serialize custom types needed for async inference
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

_WORKER_UPDATE_TIMING_KEYS = ("deserialize", "convert", "sync_weights", "total")


def serialize_state_dict_for_rpc(state_dict: dict) -> dict:
    """Serialize numpy arrays to (bytes, dtype, shape) tuples for RPC transfer.

    vLLM's collective_rpc can corrupt numpy arrays during serialization.
    This converts them to a format that survives pickling.
    """
    serialized = {}
    for key, value in state_dict.items():
        if isinstance(value, np.ndarray):
            serialized[key] = (value.tobytes(), str(value.dtype), value.shape)
        else:
            # Already serializable (or will fail later with a clear error)
            serialized[key] = value
    return serialized


def _summarize_worker_update_metrics(worker_results: Any) -> dict[str, float | int]:
    if isinstance(worker_results, Mapping):
        worker_records = [worker_results]
    elif isinstance(worker_results, (list, tuple)):
        worker_records = [record for record in worker_results if isinstance(record, Mapping)]
    else:
        worker_records = []

    if not worker_records:
        return {}

    summary: dict[str, float | int] = {
        "vllm_inflight/worker_count": len(worker_records),
    }
    for metric_name in _WORKER_UPDATE_TIMING_KEYS:
        values = [
            float(value)
            for record in worker_records
            if isinstance((value := record.get(metric_name)), Real) and not isinstance(value, bool)
        ]
        if values:
            summary[f"vllm_inflight/worker_{metric_name}_avg"] = sum(values) / len(values)
            summary[f"vllm_inflight/worker_{metric_name}_max"] = max(values)

    param_counts = [
        int(value)
        for record in worker_records
        if isinstance((value := record.get("param_count")), Real) and not isinstance(value, bool)
    ]
    if param_counts:
        summary["vllm_inflight/worker_param_count_max"] = max(param_counts)

    return summary


class AsyncvLLMInferenceContext(vLLMInferenceContext):
    """Inference context for async vLLM."""

    def __init__(self, inference_config: vLLMInferenceContextConfig):
        super().__init__(
            dataclasses.replace(
                inference_config,
                engine=dataclasses.replace(inference_config.engine, mode=InferenceMode.ASYNC),
            )
        )
        self._update_count = 0
        self._last_update_metrics: dict[str, float | int] = {}

    def reload_model(self, model: LmHeadModel | None, state_dict: dict) -> None:
        start_time = time.time()
        # Serialize numpy arrays to (bytes, dtype, shape) tuples to survive RPC serialization.
        # vLLM's collective_rpc can corrupt numpy arrays during pickling.
        serialized_state_dict = serialize_state_dict_for_rpc(state_dict)
        serialize_done = time.time()
        worker_update_metrics = self.llm.update_weights(serialized_state_dict, self.canonical_model_name)
        update_done = time.time()
        self.llm.reset_prefix_cache()  # Reset prefix cache because of new weights
        reset_done = time.time()
        self._update_count += 1
        self._last_update_metrics = {
            "vllm_inflight/update_count": self._update_count,
            "vllm_inflight/param_count": len(state_dict),
            "vllm_inflight/serialize": serialize_done - start_time,
            "vllm_inflight/rpc_update": update_done - serialize_done,
            "vllm_inflight/prefix_cache_reset": reset_done - update_done,
            "vllm_inflight/update_total": reset_done - start_time,
        }
        self._last_update_metrics.update(_summarize_worker_update_metrics(worker_update_metrics))
        logger.info(
            "Async vLLM weight update complete: params=%d serialize=%.2fs rpc=%.2fs reset=%.2fs total=%.2fs",
            len(state_dict),
            self._last_update_metrics["vllm_inflight/serialize"],
            self._last_update_metrics["vllm_inflight/rpc_update"],
            self._last_update_metrics["vllm_inflight/prefix_cache_reset"],
            self._last_update_metrics["vllm_inflight/update_total"],
        )

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._last_update_metrics)

    def shutdown(self):
        self.llm.shutdown()

    def start_server(self, model: LmHeadModel) -> None:
        pass
