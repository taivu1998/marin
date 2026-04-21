# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment wrapper for Prime Intellect verifier environments."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Mapping
from typing import Any, ClassVar

from marin.rl.environments import MarinEnv
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

_SUPPORTED_OWNER = "primeintellect"
_ENV_NAME_PREFIX = "prime_intellect:"


class PrimeIntellectEnv(MarinEnv):
    """Adapter for Phase 1 Prime Intellect verifier environments."""

    INSTALLED_ENV_IDS: ClassVar[set[str]] = set()
    LOADED_ENVIRONMENTS: ClassVar[dict[tuple[str, object], Any]] = {}

    def __init__(
        self,
        env_id: str,
        env_args: dict[str, object] | None = None,
        max_tokens: int = 1024,
        max_concurrent: int = 32,
    ):
        self.env_id = env_id
        self.env_args = dict(env_args or {})
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self._normalized_env_args = freeze_cache_value(self.env_args)
        self._is_prepared = False

    def environment_identity(self) -> EnvironmentIdentity:
        return EnvironmentIdentity(
            task_name=f"{_ENV_NAME_PREFIX}{self.env_id}",
            task_version="v1",
            verifier_name=f"verifiers:{self.env_id}",
            verifier_version="v1",
        )

    def _verifiers_module(self) -> Any:
        return import_verifiers(
            "The 'verifiers' package is required to use PrimeIntellectEnv. " "Please install it with: uv sync --extra rl"
        )

    def _short_env_id(self) -> str:
        owner, separator, slug = self.env_id.partition("/")
        if owner != _SUPPORTED_OWNER or separator == "" or not slug:
            raise ValueError(f"PrimeIntellectEnv Phase 1 only supports '{_SUPPORTED_OWNER}/*' IDs, got {self.env_id!r}")
        return slug

    def _verifier_cache_key(self) -> tuple[str, object]:
        return self.env_id, self._normalized_env_args

    def prepare(self) -> None:
        if self._is_prepared:
            return None

        self._verifiers_module()
        self._short_env_id()

        prime_executable = shutil.which("prime")
        if prime_executable is None:
            raise RuntimeError(
                "PrimeIntellectEnv requires the 'prime' executable on PATH. "
                "Install the Prime CLI before running Prime verifier environments."
            )

        if self.env_id not in self.INSTALLED_ENV_IDS:
            subprocess.run([prime_executable, "env", "install", self.env_id], check=True)
            self.INSTALLED_ENV_IDS.add(self.env_id)

        self._is_prepared = True
        return None

    def _load_verifier_env(self) -> Any:
        cache_key = self._verifier_cache_key()
        if cache_key in self.LOADED_ENVIRONMENTS:
            return self.LOADED_ENVIRONMENTS[cache_key]

        vf = self._verifiers_module()
        verifier_env = vf.load_environment(env_id=self._short_env_id(), **self.env_args)
        self.LOADED_ENVIRONMENTS[cache_key] = verifier_env
        return verifier_env

    def _validate_sample_request(self, mode: str, system_prompt: str | None) -> None:
        if not self._is_prepared:
            raise RuntimeError("PrimeIntellectEnv.prepare() must be called before sample()")
        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported mode: {mode!r}")
        if system_prompt is not None:
            raise ValueError("PrimeIntellectEnv Phase 1 does not support Marin-level system prompts")

    def _validate_verifier_env(self, verifier_env: Any) -> None:
        message_type = getattr(verifier_env, "message_type", None)
        if message_type != "chat":
            raise ValueError(
                f"PrimeIntellectEnv Phase 1 only supports chat-format verifier environments, got {message_type!r}"
            )

        tool_defs = getattr(verifier_env, "tool_defs", None)
        if tool_defs:
            raise ValueError("PrimeIntellectEnv Phase 1 does not support tool-enabled verifier environments")

    def _select_inputs(self, verifier_env: Any, mode: str, n_examples: int) -> Any:
        if mode == "train":
            return verifier_env.get_dataset(n=n_examples)
        return verifier_env.get_eval_dataset(n=n_examples)

    def _repeat_inputs(self, inputs: Any, n_generations: int) -> Any:
        if n_generations == 1:
            return inputs
        if hasattr(inputs, "repeat"):
            return inputs.repeat(n_generations)
        raise TypeError("PrimeIntellectEnv expects verifier datasets to expose repeat()")

    def _extract_example_ids(self, inputs: Any) -> list[str]:
        if hasattr(inputs, "column_names") and "id" in inputs.column_names:
            return [str(example_id) for example_id in inputs["id"]]
        return [f"example_{index}" for index in range(len(inputs))]

    def _scalarize_metrics(self, raw_metrics: Mapping[str, object]) -> dict[str, float]:
        metrics = {}
        for metric_name, values in raw_metrics.items():
            metrics[f"{self.env_id}.{metric_name}"] = scalarize_metric(metric_name, values)
        return metrics

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
        del prng_key

        self._validate_sample_request(mode, system_prompt)
        verifier_env = self._load_verifier_env()
        self._validate_verifier_env(verifier_env)

        base_inputs = self._select_inputs(verifier_env, mode, n_examples)
        if base_inputs is None:
            raise ValueError(f"PrimeIntellectEnv could not load any inputs for mode {mode!r}")

        example_ids = self._extract_example_ids(base_inputs)
        repeated_inputs = self._repeat_inputs(base_inputs, n_generations)
        expected_rollouts = len(example_ids) * n_generations

        sampling_args = {
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "logprobs": True,
            "stop": stop,
        }

        result = verifier_env.generate(
            inputs=repeated_inputs,
            client=inference_ctx.openai_client(),
            model="marin-model",
            sampling_args=sampling_args,
            max_concurrent=self.max_concurrent,
        )
        validate_generate_outputs(result, expected_rollouts)

        raw_metrics = getattr(result, "metrics", {}) or {}
        metrics = self._scalarize_metrics(raw_metrics)
        identity = self.environment_identity()

        if expected_rollouts == 0:
            metrics[f"{self.env_id}.total_rollouts"] = 0.0
            return EnvironmentSample(rollout_groups=[], metrics=metrics, traces=[], identity=identity)

        rollout_groups = []
        traces = []
        reward_sum = 0.0
        n_sampled_examples = len(example_ids)

        for example_index, example_id in enumerate(example_ids):
            group_rollouts = []
            response_traces = []
            trace_prompt: list[dict[str, str]] | None = None
            env_example_id = f"{self.env_id}:{example_id}"

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
                rollout = inference_ctx.create_rollout_from_choice(
                    prompt=prompt_messages,
                    choice=choice,
                    env_name=f"{_ENV_NAME_PREFIX}{self.env_id}",
                    env_example_id=env_example_id,
                    reward=reward,
                    temperature=temperature,
                    top_k=top_k,
                )
                group_rollouts.append(rollout)
                reward_sum += reward
                response_traces.append(
                    EpisodeResponseTrace(
                        response_text=response_text,
                        reward=reward,
                        correctness_reward=rollout.correctness_reward,
                        is_truncated=rollout.is_truncated,
                    )
                )

            rollout_groups.append(RolloutGroup(rollouts=group_rollouts))
            traces.append(
                EpisodeTrace(
                    env_name=f"{_ENV_NAME_PREFIX}{self.env_id}",
                    env_example_id=env_example_id,
                    prompt=trace_prompt or "",
                    responses=tuple(response_traces),
                    task_name=identity.task_name,
                    task_version=identity.task_version,
                    verifier_name=identity.verifier_name,
                    verifier_version=identity.verifier_version,
                )
            )

        metrics[f"{self.env_id}.mean_reward"] = reward_sum / expected_rollouts
        metrics[f"{self.env_id}.total_rollouts"] = float(expected_rollouts)
        logger.info("Generated %d rollout groups for %s", len(rollout_groups), self.env_id)
        return EnvironmentSample(rollout_groups=rollout_groups, metrics=metrics, traces=traces, identity=identity)
