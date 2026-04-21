# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for environment loading from EnvConfig."""

from marin.rl.environments import EnvConfig, load_environment_from_spec
from marin.rl.environments.math_env import MathEnv
from marin.rl.environments.mock_env import MockEnv
from marin.rl.environments.prime_reasoning_gym_env import PrimeReasoningGymEnv


def test_load_mock_environment():
    """Test loading MockEnv via EnvConfig."""
    config = EnvConfig(env_class="marin.rl.environments.mock_env.MockEnv", env_args={"task_type": "cats", "seed": 42})

    env = load_environment_from_spec(config)

    assert isinstance(env, MockEnv)
    assert env.task_type == "cats"
    assert len(env.train_examples) > 0
    assert len(env.eval_examples) > 0
    assert env.environment_identity().task_name == "mock_env:cats"


def test_load_math_environment():
    """Test loading MathEnv via EnvConfig with inline data (no HF download)."""
    config = EnvConfig(
        env_class="marin.rl.environments.math_env.MathEnv",
        env_args={
            "seed": 42,
            "train_dataset": [{"problem": "What is 1+1?", "solution": "\\boxed{2}"}],
            "eval_dataset": [{"problem": "What is 2+2?", "solution": "\\boxed{4}"}],
        },
    )

    env = load_environment_from_spec(config)

    assert isinstance(env, MathEnv)
    assert len(env.train_examples) > 0
    assert len(env.eval_examples) > 0
    assert env.environment_identity().task_name == "math"


def test_load_prime_reasoning_gym_environment():
    """Test loading PrimeReasoningGymEnv via EnvConfig without preparing dependencies."""
    config = EnvConfig(
        env_class="marin.rl.environments.prime_reasoning_gym_env.PrimeReasoningGymEnv",
        env_args={
            "gym": "leg_counting",
            "num_train_examples": 10,
            "num_eval_examples": 4,
            "seed": 42,
        },
    )

    env = load_environment_from_spec(config)

    assert isinstance(env, PrimeReasoningGymEnv)
    assert env.gym == "leg_counting"
    assert env.num_train_examples == 10
    assert env.num_eval_examples == 4
    assert env.environment_identity().task_name == "prime_reasoning_gym:leg_counting"
