# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prime/verifiers-backed Reasoning Gym environment for Marin RL."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from marin.rl.environments import MarinEnv
from marin.rl.environments.base import extract_seed
from marin.rl.environments.inference_ctx import BaseInferenceContext
from marin.rl.environments.spec import EnvironmentIdentity, EnvironmentSample
from marin.rl.environments.verifiers_support import (
    extract_single_turn_chat_rollout,
    freeze_cache_value,
    import_verifiers,
    scalarize_metric,
    validate_generate_outputs,
)
from marin.rl.traces import EpisodeResponseTrace, EpisodeTrace
from marin.rl.types import RolloutGroup

logger = logging.getLogger(__name__)

_ENV_NAME_PREFIX = "prime_reasoning_gym"
_METRIC_FRAGMENT_PATTERN = re.compile(r"[^A-Za-z0-9_]")
_DEFAULT_SUCCESS_THRESHOLD = 1.0


def _jsonable_gym_config(config: object) -> object:
    if isinstance(config, (str, int, float, bool, type(None))):
        return config
    if isinstance(config, Sequence) and not isinstance(config, (str, bytes, bytearray)):
        return [_jsonable_gym_config(item) for item in config]
    if isinstance(config, Mapping):
        normalized = {}
        for key, value in config.items():
            if not isinstance(key, str):
                raise TypeError(f"Reasoning Gym config keys must be strings, got {type(key).__name__}")
            normalized[key] = _jsonable_gym_config(value)
        return normalized
    raise TypeError(f"Unsupported Reasoning Gym config value type: {type(config).__name__}")


def _metric_fragment(value: str) -> str:
    return _METRIC_FRAGMENT_PATTERN.sub("_", value)


def _gym_label(gym: object) -> str:
    if isinstance(gym, str):
        return _metric_fragment(gym)

    serialized = json.dumps(_jsonable_gym_config(gym), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]
    return f"composite_{digest}"


class PrimeReasoningGymEnv(MarinEnv):
    """Thin Marin adapter over `verifiers.ReasoningGymEnv`."""

    def __init__(
        self,
        gym: str | list[str | dict[str, object]],
        num_train_examples: int = 1000,
        num_eval_examples: int = 100,
        seed: int = 0,
        success_threshold: float = _DEFAULT_SUCCESS_THRESHOLD,
        max_tokens: int = 1024,
        max_concurrent: int = 32,
        system_prompt: str | None = None,
    ) -> None:
        self.gym = gym
        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples
        self.seed = seed
        self.success_threshold = success_threshold
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.system_prompt = system_prompt
        self._vf_env: Any | None = None
        self._is_prepared = False
        self._label = _gym_label(gym)
        self._env_name = f"{_ENV_NAME_PREFIX}:{self._label}"
        self._normalized_gym = freeze_cache_value(_jsonable_gym_config(gym))

    def environment_identity(self) -> EnvironmentIdentity:
        return EnvironmentIdentity(
            task_name=self._env_name,
            task_version="v1",
            verifier_name="verifiers:reasoning_gym",
            verifier_version="v1",
        )

    def _verifiers_module(self) -> Any:
        return import_verifiers(
            "The 'verifiers' package is required to use PrimeReasoningGymEnv. "
            "Please install it with: uv sync --extra rl --extra reasoning-gym"
        )

    def _ensure_reasoning_gym_installed(self) -> None:
        try:
            importlib.import_module("reasoning_gym")
        except ModuleNotFoundError as exc:
            if exc.name != "reasoning_gym":
                raise
            raise ImportError(
                "The 'reasoning_gym' package is required to use PrimeReasoningGymEnv. "
                "Please install it with: uv sync --extra reasoning-gym"
            ) from exc

    def prepare(self) -> None:
        if self._is_prepared:
            return None

        self._ensure_reasoning_gym_installed()
        vf = self._verifiers_module()
        if not hasattr(vf, "ReasoningGymEnv"):
            raise ImportError(
                "Installed 'verifiers' package does not expose ReasoningGymEnv. "
                "Resync the rl extra or install a verifiers build that includes ReasoningGymEnv."
            )

        init_kwargs = {
            "gym": self.gym,
            "num_train_examples": self.num_train_examples,
            "num_eval_examples": self.num_eval_examples,
            "seed": self.seed,
        }
        if self.system_prompt is not None:
            init_kwargs["system_prompt"] = self.system_prompt

        self._vf_env = vf.ReasoningGymEnv(**init_kwargs)
        self._is_prepared = True
        return None

    def _require_prepared_env(self) -> Any:
        if not self._is_prepared or self._vf_env is None:
            raise RuntimeError("PrimeReasoningGymEnv.prepare() must be called before sample()")
        return self._vf_env

    def _base_inputs(self, mode: str):
        vf_env = self._require_prepared_env()
        if mode == "train":
            dataset = vf_env.dataset
        elif mode == "eval":
            dataset = vf_env.eval_dataset
        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

        if dataset is None:
            raise ValueError(f"No dataset available for mode {mode!r}")
        return dataset

    def _sample_inputs(self, mode: str, n_examples: int, prng_key) -> Any:
        base_inputs = self._base_inputs(mode)
        sample_size = min(n_examples, len(base_inputs))
        shuffled = base_inputs.shuffle(seed=extract_seed(prng_key))
        return shuffled.select(range(sample_size))

    def _metrics_prefix(self, mode: str) -> str:
        return f"{_ENV_NAME_PREFIX}.{self._label}.{mode}"

    def _scalarize_metrics(self, raw_metrics: Mapping[str, object], mode: str) -> dict[str, float]:
        prefix = self._metrics_prefix(mode)
        metrics = {}
        for metric_name, values in raw_metrics.items():
            metrics[f"{prefix}_{_metric_fragment(metric_name)}"] = scalarize_metric(metric_name, values)
        return metrics

    def _build_example_id(self, row: Mapping[str, object]) -> str:
        task_name = row.get("task")
        if not isinstance(task_name, str) or not task_name:
            task_name = self._label

        answer_index = row.get("answer")
        if not isinstance(answer_index, str) or not answer_index:
            raise ValueError("Reasoning Gym verifier dataset rows must contain a string 'answer' index")

        return f"reasoning_gym:{task_name}:{answer_index}"

    def sample(
        self,
        inference_ctx: BaseInferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
        max_tokens: int | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> EnvironmentSample:
        if n_examples <= 0:
            raise ValueError("n_examples must be positive")
        if n_generations <= 0:
            raise ValueError("n_generations must be positive")
        if system_prompt is not None:
            raise ValueError(
                "PrimeReasoningGymEnv does not support Marin-level system prompts. "
                "Set system_prompt on the environment config instead."
            )

        vf_env = self._require_prepared_env()
        sampled_inputs = self._sample_inputs(mode, n_examples, prng_key)
        sampled_rows = sampled_inputs.to_list()
        repeated_inputs = sampled_inputs.repeat(n_generations) if n_generations > 1 else sampled_inputs
        expected_rollouts = len(sampled_rows) * n_generations

        sampling_args = {
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "logprobs": True,
            "stop": stop,
        }
        result = vf_env.generate(
            inputs=repeated_inputs,
            client=inference_ctx.openai_client(),
            model="marin-model",
            sampling_args=sampling_args,
            max_concurrent=self.max_concurrent,
        )
        validate_generate_outputs(result, expected_rollouts)

        identity = self.environment_identity()
        raw_metrics = getattr(result, "metrics", {}) or {}
        metrics = self._scalarize_metrics(raw_metrics, mode)

        if expected_rollouts == 0:
            metrics[f"{self._metrics_prefix(mode)}_total_responses"] = 0.0
            return EnvironmentSample(rollout_groups=[], metrics=metrics, traces=[], identity=identity)

        rollout_groups = []
        traces = []
        reward_sum = 0.0
        solve_sum = 0.0
        response_token_count = 0
        truncated_count = 0
        source_counts: dict[str, int] = {}
        n_sampled_examples = len(sampled_rows)

        for example_index, row in enumerate(sampled_rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"Reasoning Gym sampled row must be a mapping, got {type(row).__name__}")

            task_name = row.get("task")
            if not isinstance(task_name, str) or not task_name:
                task_name = self._label
            source_counts[task_name] = source_counts.get(task_name, 0) + 1

            env_example_id = self._build_example_id(row)
            group_rollouts = []
            response_traces = []
            trace_prompt: list[dict[str, str]] | None = None

            for generation_index in range(n_generations):
                rollout_index = generation_index * n_sampled_examples + example_index
                prompt_messages, choice, response_text = extract_single_turn_chat_rollout(
                    rollout_index=rollout_index,
                    prompt=result.prompt[rollout_index],
                    completion=result.completion[rollout_index],
                    state=result.state[rollout_index],
                )
                if trace_prompt is None:
                    trace_prompt = prompt_messages

                reward = float(result.reward[rollout_index])
                solved = float(reward >= self.success_threshold)
                rollout = inference_ctx.create_rollout_from_choice(
                    prompt=prompt_messages,
                    choice=choice,
                    env_name=self._env_name,
                    env_example_id=env_example_id,
                    reward=reward,
                    correctness_reward=solved,
                    temperature=temperature,
                    top_k=top_k,
                )
                group_rollouts.append(rollout)
                reward_sum += reward
                solve_sum += solved
                response_token_count += rollout.response_tokens.size
                if rollout.is_truncated:
                    truncated_count += 1
                response_traces.append(
                    EpisodeResponseTrace(
                        response_text=response_text,
                        reward=reward,
                        correctness_reward=solved,
                        is_truncated=rollout.is_truncated,
                    )
                )

            rollout_groups.append(RolloutGroup(rollouts=group_rollouts))
            traces.append(
                EpisodeTrace(
                    env_name=self._env_name,
                    env_example_id=env_example_id,
                    prompt=trace_prompt or "",
                    responses=tuple(response_traces),
                    task_name=identity.task_name,
                    task_version=identity.task_version,
                    verifier_name=identity.verifier_name,
                    verifier_version=identity.verifier_version,
                )
            )

        prefix = self._metrics_prefix(mode)
        total_choices = float(expected_rollouts)
        metrics[f"{prefix}_mean_reward"] = reward_sum / total_choices
        metrics[f"{prefix}_solve_rate"] = solve_sum / total_choices
        metrics[f"{prefix}_mean_response_tokens"] = response_token_count / total_choices
        metrics[f"{prefix}_total_responses"] = total_choices
        metrics[f"{prefix}_sampled_examples"] = float(len(sampled_rows))
        metrics[f"{prefix}_truncated_percentage"] = float(truncated_count) / total_choices
        for source_name, count in sorted(source_counts.items()):
            metrics[f"{prefix}_source_{_metric_fragment(source_name)}_count"] = float(count)

        return EnvironmentSample(rollout_groups=rollout_groups, metrics=metrics, traces=traces, identity=identity)
