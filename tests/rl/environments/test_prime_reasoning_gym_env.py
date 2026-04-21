# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Prime/verifiers-backed Reasoning Gym environment adapter."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType

import jax.random
import pytest
from datasets import Dataset
from marin.rl.environments.prime_reasoning_gym_env import PrimeReasoningGymEnv

from tests.rl.environments.verifiers_test_support import (
    install_fake_reasoning_gym,
    install_fake_verifiers,
    levanter_inference_ctx,
    single_turn_generate_outputs,
)


def _reasoning_gym_env_cls(
    gpt2_tokenizer,
    *,
    train_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
    created_kwargs: list[dict[str, object]],
    reward_by_answer: dict[str, list[float]] | None = None,
    response_by_answer: dict[str, list[str]] | None = None,
):
    class FakeReasoningGymVerifierEnv:
        def __init__(self, **kwargs):
            created_kwargs.append(dict(kwargs))
            self.dataset = Dataset.from_list(train_rows)
            self.eval_dataset = Dataset.from_list(eval_rows)
            self.generate_calls: list[dict[str, object]] = []

        def generate(self, *, inputs, client, model, sampling_args, max_concurrent):
            self.generate_calls.append(
                {
                    "inputs": inputs,
                    "client": client,
                    "model": model,
                    "sampling_args": dict(sampling_args),
                    "max_concurrent": max_concurrent,
                }
            )
            return single_turn_generate_outputs(
                inputs,
                gpt2_tokenizer,
                reward_by_answer=reward_by_answer,
                response_by_answer=response_by_answer,
                metric_scale=4.0,
            )

    return FakeReasoningGymVerifierEnv


def test_prime_reasoning_gym_env_sample_uses_scores_and_binary_correctness(monkeypatch, gpt2_tokenizer):
    created_kwargs: list[dict[str, object]] = []
    fake_rg_cls = _reasoning_gym_env_cls(
        gpt2_tokenizer,
        train_rows=[
            {"question": "How many legs?", "answer": "0", "task": "leg_counting"},
        ],
        eval_rows=[
            {"question": "Eval question", "answer": "1", "task": "leg_counting"},
        ],
        created_kwargs=created_kwargs,
        reward_by_answer={"0": [1.0, 0.25]},
        response_by_answer={"0": ["4", "5"]},
    )
    install_fake_reasoning_gym(monkeypatch)
    install_fake_verifiers(monkeypatch, reasoning_gym_env_cls=fake_rg_cls)

    env = PrimeReasoningGymEnv(
        gym="leg_counting",
        num_train_examples=1,
        num_eval_examples=1,
        seed=7,
        success_threshold=1.0,
        max_tokens=128,
        max_concurrent=5,
    )
    inference_ctx = levanter_inference_ctx(gpt2_tokenizer)

    env.prepare()
    sample = env.sample(
        inference_ctx=inference_ctx,
        n_examples=1,
        n_generations=2,
        temperature=0.7,
        prng_key=jax.random.PRNGKey(0),
        mode="train",
    )

    assert created_kwargs == [
        {
            "gym": "leg_counting",
            "num_train_examples": 1,
            "num_eval_examples": 1,
            "seed": 7,
        }
    ]
    assert sample.identity.task_name == "prime_reasoning_gym:leg_counting"
    assert sample.identity.verifier_name == "verifiers:reasoning_gym"
    assert len(sample.rollout_groups) == 1
    assert len(sample.rollout_groups[0].rollouts) == 2

    correct_rollout, partial_rollout = sample.rollout_groups[0].rollouts
    assert correct_rollout.env_name == "prime_reasoning_gym:leg_counting"
    assert correct_rollout.env_example_id == "reasoning_gym:leg_counting:0"
    assert correct_rollout.episode_reward == pytest.approx(1.0)
    assert correct_rollout.correctness_reward == pytest.approx(1.0)
    assert partial_rollout.episode_reward == pytest.approx(0.25)
    assert partial_rollout.correctness_reward == pytest.approx(0.0)
    assert sample.traces[0].responses[0].correctness_reward == pytest.approx(1.0)
    assert sample.traces[0].responses[1].correctness_reward == pytest.approx(0.0)
    assert sample.metrics == {
        "prime_reasoning_gym.leg_counting.train_score": pytest.approx(2.5),
        "prime_reasoning_gym.leg_counting.train_mean_reward": pytest.approx(0.625),
        "prime_reasoning_gym.leg_counting.train_solve_rate": pytest.approx(0.5),
        "prime_reasoning_gym.leg_counting.train_mean_response_tokens": pytest.approx(1.0),
        "prime_reasoning_gym.leg_counting.train_total_responses": 2.0,
        "prime_reasoning_gym.leg_counting.train_sampled_examples": 1.0,
        "prime_reasoning_gym.leg_counting.train_truncated_percentage": 0.0,
        "prime_reasoning_gym.leg_counting.train_source_leg_counting_count": 1.0,
    }


def test_prime_reasoning_gym_env_sampling_is_deterministic_for_fixed_prng_key(monkeypatch, gpt2_tokenizer):
    created_kwargs: list[dict[str, object]] = []
    fake_rg_cls = _reasoning_gym_env_cls(
        gpt2_tokenizer,
        train_rows=[
            {"question": "Q0", "answer": "0", "task": "leg_counting"},
            {"question": "Q1", "answer": "1", "task": "leg_counting"},
            {"question": "Q2", "answer": "2", "task": "leg_counting"},
        ],
        eval_rows=[
            {"question": "Eval", "answer": "9", "task": "leg_counting"},
        ],
        created_kwargs=created_kwargs,
        reward_by_answer={"0": [1.0], "1": [1.0], "2": [1.0]},
        response_by_answer={"0": ["A0"], "1": ["A1"], "2": ["A2"]},
    )
    install_fake_reasoning_gym(monkeypatch)
    install_fake_verifiers(monkeypatch, reasoning_gym_env_cls=fake_rg_cls)

    env = PrimeReasoningGymEnv(gym="leg_counting", num_train_examples=3, num_eval_examples=1, seed=10)
    env.prepare()
    inference_ctx = levanter_inference_ctx(gpt2_tokenizer)
    prng_key = jax.random.PRNGKey(123)

    first_sample = env.sample(
        inference_ctx=inference_ctx,
        n_examples=2,
        n_generations=1,
        temperature=1.0,
        prng_key=prng_key,
        mode="train",
    )
    second_sample = env.sample(
        inference_ctx=inference_ctx,
        n_examples=2,
        n_generations=1,
        temperature=1.0,
        prng_key=prng_key,
        mode="train",
    )

    first_ids = [group.rollouts[0].env_example_id for group in first_sample.rollout_groups]
    second_ids = [group.rollouts[0].env_example_id for group in second_sample.rollout_groups]
    assert first_ids == second_ids


def test_prime_reasoning_gym_env_supports_composite_gym_configs(monkeypatch, gpt2_tokenizer):
    created_kwargs: list[dict[str, object]] = []
    fake_rg_cls = _reasoning_gym_env_cls(
        gpt2_tokenizer,
        train_rows=[
            {"question": "Composite question", "answer": "11", "task": "tower_of_hanoi"},
        ],
        eval_rows=[
            {"question": "Composite eval question", "answer": "12", "task": "leg_counting"},
        ],
        created_kwargs=created_kwargs,
        reward_by_answer={"11": [1.0]},
        response_by_answer={"11": ["ABC"]},
    )
    install_fake_reasoning_gym(monkeypatch)
    install_fake_verifiers(monkeypatch, reasoning_gym_env_cls=fake_rg_cls)

    gym_config = [
        {"name": "tower_of_hanoi", "weight": 1.0, "config": {"min_disks": 3, "max_disks": 4}},
        {"name": "leg_counting", "weight": 1.0, "config": {"min_animals": 2, "max_animals": 3}},
    ]
    env = PrimeReasoningGymEnv(
        gym=gym_config,
        num_train_examples=1,
        num_eval_examples=1,
        seed=4,
    )

    env.prepare()
    sample = env.sample(
        inference_ctx=levanter_inference_ctx(gpt2_tokenizer),
        n_examples=1,
        n_generations=1,
        temperature=1.0,
        prng_key=jax.random.PRNGKey(1),
        mode="train",
    )

    assert created_kwargs[0]["gym"] == gym_config
    assert sample.identity.task_name.startswith("prime_reasoning_gym:composite_")
    assert sample.rollout_groups[0].rollouts[0].env_example_id == "reasoning_gym:tower_of_hanoi:11"
    composite_prefix = sample.identity.task_name.replace(":", ".")
    assert sample.metrics[f"{composite_prefix}.train_source_tower_of_hanoi_count"] == pytest.approx(1.0)


def test_prime_reasoning_gym_env_requires_reasoning_gym_dependency(monkeypatch):
    fake_verifiers = ModuleType("verifiers")
    fake_verifiers.ReasoningGymEnv = object
    monkeypatch.setitem(sys.modules, "verifiers", fake_verifiers)

    module = importlib.import_module("marin.rl.environments.prime_reasoning_gym_env")
    real_import_module = module.importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "reasoning_gym":
            error = ModuleNotFoundError("No module named 'reasoning_gym'")
            error.name = "reasoning_gym"
            raise error
        return real_import_module(name, package)

    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)
    env = PrimeReasoningGymEnv(gym="leg_counting")

    with pytest.raises(ImportError, match="uv sync --extra reasoning-gym"):
        env.prepare()


def test_prime_reasoning_gym_env_requires_verifiers_reasoning_gym_env(monkeypatch):
    install_fake_reasoning_gym(monkeypatch)
    install_fake_verifiers(monkeypatch)
    env = PrimeReasoningGymEnv(gym="leg_counting")

    with pytest.raises(ImportError, match="does not expose ReasoningGymEnv"):
        env.prepare()


def test_prime_reasoning_gym_env_requires_real_verifiers_reasoning_gym_env_when_installed():
    if importlib.util.find_spec("verifiers") is None or importlib.util.find_spec("reasoning_gym") is None:
        pytest.skip("verifiers and reasoning_gym are not installed in this environment")

    verifiers = importlib.import_module("verifiers")
    assert hasattr(verifiers, "ReasoningGymEnv")


def test_prime_reasoning_gym_env_rejects_runtime_system_prompt(monkeypatch, gpt2_tokenizer):
    created_kwargs: list[dict[str, object]] = []
    fake_rg_cls = _reasoning_gym_env_cls(
        gpt2_tokenizer,
        train_rows=[{"question": "Q0", "answer": "0", "task": "leg_counting"}],
        eval_rows=[{"question": "Eval", "answer": "1", "task": "leg_counting"}],
        created_kwargs=created_kwargs,
        reward_by_answer={"0": [1.0]},
        response_by_answer={"0": ["A0"]},
    )
    install_fake_reasoning_gym(monkeypatch)
    install_fake_verifiers(monkeypatch, reasoning_gym_env_cls=fake_rg_cls)
    env = PrimeReasoningGymEnv(gym="leg_counting")
    env.prepare()

    with pytest.raises(ValueError, match="does not support Marin-level system prompts"):
        env.sample(
            inference_ctx=levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=jax.random.PRNGKey(0),
            system_prompt="override",
        )
