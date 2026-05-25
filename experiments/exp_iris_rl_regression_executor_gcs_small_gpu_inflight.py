# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun because vLLM is not installed by default

"""Regression probe: executor + small GPU-only RL launch with in-flight vLLM updates."""

import argparse
import dataclasses
import datetime
import logging
import os

from marin.execution.executor import executor_main
from marin.rl.rl_experiment_utils import RLExperimentConfig, executor_main_config_for_rl_experiment, make_rl_step

from experiments.exp_iris_rl_regression_executor_gcs_small_gpu import build_debug_config
from experiments.iris_rl_gpu_smoke import (
    DEFAULT_EXPERIMENT_REGION,
    DEFAULT_GPU_COUNT,
    DEFAULT_GPU_TYPE,
    gpu_smoke_curriculum,
)

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_SUFFIX = "exec-gcs-small-gpu-inflight"
DEFAULT_INFLIGHT_NUM_TRAIN_STEPS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name-suffix",
        default=DEFAULT_EXPERIMENT_SUFFIX,
        help="Run-name suffix used for job and W&B labeling.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=DEFAULT_INFLIGHT_NUM_TRAIN_STEPS,
        help="Number of RL training steps to execute.",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_EXPERIMENT_REGION,
        help="Region for trainer and rollout GPU jobs.",
    )
    parser.add_argument(
        "--gpu-type",
        default=DEFAULT_GPU_TYPE,
        help="GPU type shared by trainer and rollout workers.",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=DEFAULT_GPU_COUNT,
        help="Number of GPUs per trainer and rollout worker host.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Model artifact path or Hugging Face model id. Defaults to the executor-managed canonical artifact.",
    )
    return parser.parse_args()


def build_inflight_debug_config(
    *,
    experiment_name_suffix: str,
    num_train_steps: int,
    region: str,
    gpu_type: str,
    gpu_count: int,
    model_path: str | None,
) -> RLExperimentConfig:
    sync_config = build_debug_config(
        experiment_name_suffix=experiment_name_suffix,
        num_train_steps=num_train_steps,
        region=region,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        model_path=model_path,
    )
    return dataclasses.replace(
        sync_config,
        inflight_weight_updates=True,
        tags=[*sync_config.tags, "inflight"],
    )


def main() -> None:
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    args = parse_args()
    debug_config = build_inflight_debug_config(
        experiment_name_suffix=args.experiment_name_suffix,
        num_train_steps=args.num_train_steps,
        region=args.region,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        model_path=args.model_path,
    )

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{args.experiment_name_suffix}-{datestamp}"
    curriculum = gpu_smoke_curriculum(
        run_id=name,
        max_input_tokens=debug_config.max_input_tokens,
        max_output_tokens=debug_config.max_output_tokens,
        num_generations=debug_config.n_generations_per_prompt,
    )
    step = make_rl_step(
        name=name,
        config=debug_config,
        curriculum=curriculum,
    )

    executor_main(
        executor_main_config_for_rl_experiment(debug_config),
        steps=[step],
        description=(
            "Iris RL regression probe: executor + small GPU-only "
            f"in-flight vLLM ({args.num_train_steps} training steps)"
        ),
    )


if __name__ == "__main__":
    main()
