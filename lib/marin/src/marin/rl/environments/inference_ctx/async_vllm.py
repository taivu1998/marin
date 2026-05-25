# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
import time
from typing import Any

import numpy as np
from levanter.models.lm_model import LmHeadModel
from marin.rl.environments.inference_ctx.vllm import InferenceMode, vLLMInferenceContext, vLLMInferenceContextConfig

logger = logging.getLogger(__name__)


# Allow vLLM to serialize custom types needed for async inference
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


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
        self.llm.update_weights(serialized_state_dict, self.canonical_model_name)
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
