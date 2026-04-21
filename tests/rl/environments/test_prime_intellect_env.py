# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase 1 PrimeIntellectEnv verifier adapter."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from marin.rl.environments.inference_ctx.openai_compat import OpenAICompatClient
from marin.rl.environments.prime_intellect_env import PrimeIntellectEnv
from openai import AsyncOpenAI

from tests.rl.environments.verifiers_test_support import (
    FakeVerifierEnv,
    install_fake_verifiers,
    levanter_inference_ctx,
    prompt_dataset,
    single_turn_generate_outputs,
    vllm_inference_ctx,
)


@pytest.fixture(autouse=True)
def clear_prime_intellect_env_caches():
    PrimeIntellectEnv.INSTALLED_ENV_IDS.clear()
    PrimeIntellectEnv.LOADED_ENVIRONMENTS.clear()
    yield
    PrimeIntellectEnv.INSTALLED_ENV_IDS.clear()
    PrimeIntellectEnv.LOADED_ENVIRONMENTS.clear()


@pytest.fixture
def prime_cli(monkeypatch):
    subprocess_run = Mock()
    monkeypatch.setattr("marin.rl.environments.prime_intellect_env.shutil.which", lambda executable: "/usr/bin/prime")
    monkeypatch.setattr("marin.rl.environments.prime_intellect_env.subprocess.run", subprocess_run)
    return subprocess_run


def test_prime_intellect_env_sample_supports_levanter_single_turn_chat(monkeypatch, prime_cli, gpt2_tokenizer):
    train_dataset = prompt_dataset(["train-0", "train-1"], "train")
    eval_dataset = prompt_dataset(["eval-0", "eval-1"], "eval")
    verifier_env = FakeVerifierEnv(
        train_dataset,
        eval_dataset,
        generate_result_factory=lambda *, inputs: single_turn_generate_outputs(inputs, gpt2_tokenizer, metric_scale=5.0),
    )
    load_calls = []

    def load_environment(env_id: str, **env_args):
        load_calls.append((env_id, dict(env_args)))
        return verifier_env

    install_fake_verifiers(monkeypatch, load_environment=load_environment)
    env = PrimeIntellectEnv(
        env_id="primeintellect/gsm8k",
        env_args={"difficulty": "easy"},
        max_tokens=128,
        max_concurrent=7,
    )
    inference_ctx = levanter_inference_ctx(gpt2_tokenizer)

    env.prepare()
    sample = env.sample(
        inference_ctx=inference_ctx,
        n_examples=2,
        n_generations=2,
        temperature=0.7,
        prng_key=None,
        mode="eval",
        top_k=11,
        stop=["<stop>"],
    )

    assert prime_cli.call_count == 1
    assert prime_cli.call_args.args == (["/usr/bin/prime", "env", "install", "primeintellect/gsm8k"],)
    assert prime_cli.call_args.kwargs == {"check": True}
    assert load_calls == [("gsm8k", {"difficulty": "easy"})]
    assert isinstance(verifier_env.generate_calls[0]["client"], AsyncOpenAI)
    assert verifier_env.generate_calls[0]["model"] == "marin-model"
    assert verifier_env.generate_calls[0]["max_concurrent"] == 7
    assert verifier_env.generate_calls[0]["sampling_args"] == {
        "max_tokens": 128,
        "temperature": 0.7,
        "top_k": 11,
        "logprobs": True,
        "stop": ["<stop>"],
    }

    assert len(sample.rollout_groups) == 2
    assert [rollout.env_example_id for rollout in sample.rollout_groups[0].rollouts] == [
        "primeintellect/gsm8k:eval-0",
        "primeintellect/gsm8k:eval-0",
    ]
    assert [rollout.env_example_id for rollout in sample.rollout_groups[1].rollouts] == [
        "primeintellect/gsm8k:eval-1",
        "primeintellect/gsm8k:eval-1",
    ]
    assert [
        gpt2_tokenizer.decode(rollout.response_tokens.tolist()) for rollout in sample.rollout_groups[0].rollouts
    ] == [
        "resp-eval-0-0",
        "resp-eval-0-1",
    ]
    assert all(
        rollout.env_name == "prime_intellect:primeintellect/gsm8k"
        for group in sample.rollout_groups
        for rollout in group.rollouts
    )
    assert sample.metrics == {
        "primeintellect/gsm8k.score": pytest.approx(12.5),
        "primeintellect/gsm8k.mean_reward": pytest.approx(2.5),
        "primeintellect/gsm8k.total_rollouts": 4.0,
    }
    assert sample.identity.task_name == "prime_intellect:primeintellect/gsm8k"
    assert sample.identity.verifier_name == "verifiers:primeintellect/gsm8k"
    assert len(sample.traces) == 2
    assert sample.traces[0].env_example_id == "primeintellect/gsm8k:eval-0"
    assert sample.traces[0].responses[0].response_text == "resp-eval-0-0"


def test_prime_intellect_env_sample_supports_vllm_single_turn_chat(monkeypatch, prime_cli, gpt2_tokenizer):
    train_dataset = prompt_dataset(["0", "1"], "train")
    verifier_env = FakeVerifierEnv(
        train_dataset,
        generate_result_factory=lambda *, inputs: single_turn_generate_outputs(inputs, gpt2_tokenizer),
    )
    load_calls = []

    def load_environment(env_id: str, **env_args):
        load_calls.append((env_id, dict(env_args)))
        return verifier_env

    install_fake_verifiers(monkeypatch, load_environment=load_environment)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k", max_tokens=64)
    inference_ctx = vllm_inference_ctx(monkeypatch, gpt2_tokenizer)

    env.prepare()
    sample = env.sample(
        inference_ctx=inference_ctx,
        n_examples=2,
        n_generations=2,
        temperature=0.2,
        prng_key=None,
        mode="train",
    )

    assert load_calls == [("gsm8k", {})]
    assert isinstance(verifier_env.generate_calls[0]["client"], OpenAICompatClient)
    assert [rollout.prompt_tokens.tolist() for rollout in sample.rollout_groups[0].rollouts] == [[100, 200], [102, 202]]
    assert [rollout.prompt_tokens.tolist() for rollout in sample.rollout_groups[1].rollouts] == [[101, 201], [103, 203]]
    assert [
        gpt2_tokenizer.decode(rollout.response_tokens.tolist()) for rollout in sample.rollout_groups[0].rollouts
    ] == [
        "resp-0-0",
        "resp-0-1",
    ]
    assert [
        gpt2_tokenizer.decode(rollout.response_tokens.tolist()) for rollout in sample.rollout_groups[1].rollouts
    ] == [
        "resp-1-0",
        "resp-1-1",
    ]
    assert sample.metrics["primeintellect/gsm8k.score"] == pytest.approx(25.0)
    assert sample.metrics["primeintellect/gsm8k.mean_reward"] == pytest.approx(2.5)
    assert sample.metrics["primeintellect/gsm8k.total_rollouts"] == 4.0


def test_prime_intellect_env_prepare_installs_once_per_env_id(monkeypatch, prime_cli, gpt2_tokenizer):
    load_calls = []

    def load_environment(env_id: str, **env_args):
        load_calls.append((env_id, dict(env_args)))
        return FakeVerifierEnv(
            prompt_dataset(["0"], "train"),
            generate_result_factory=lambda *, inputs: single_turn_generate_outputs(inputs, gpt2_tokenizer),
        )

    install_fake_verifiers(monkeypatch, load_environment=load_environment)
    env_one = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={"difficulty": "easy"})
    env_two = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={"difficulty": "hard"})

    env_one.prepare()
    env_two.prepare()

    assert prime_cli.call_count == 1
    assert load_calls == []


def test_prime_intellect_env_load_cache_keys_include_env_args(monkeypatch, prime_cli, gpt2_tokenizer):
    load_calls = []

    def load_environment(env_id: str, **env_args):
        load_calls.append((env_id, dict(env_args)))
        return FakeVerifierEnv(
            prompt_dataset([str(env_args["difficulty"])], "train"),
            generate_result_factory=lambda *, inputs: single_turn_generate_outputs(inputs, gpt2_tokenizer),
        )

    install_fake_verifiers(monkeypatch, load_environment=load_environment)
    inference_ctx = levanter_inference_ctx(gpt2_tokenizer)

    easy_one = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={"difficulty": "easy"})
    easy_two = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={"difficulty": "easy"})
    hard = PrimeIntellectEnv(env_id="primeintellect/gsm8k", env_args={"difficulty": "hard"})

    for env in (easy_one, easy_two, hard):
        env.prepare()
        env.sample(
            inference_ctx=inference_ctx,
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
        )

    assert load_calls == [
        ("gsm8k", {"difficulty": "easy"}),
        ("gsm8k", {"difficulty": "hard"}),
    ]


def test_prime_intellect_env_prepare_rejects_non_primeintellect_ids(monkeypatch, prime_cli):
    install_fake_verifiers(monkeypatch, load_environment=lambda env_id, **env_args: None)
    env = PrimeIntellectEnv(env_id="someone-else/gsm8k")

    with pytest.raises(ValueError, match="only supports 'primeintellect/\\*' IDs"):
        env.prepare()

    assert prime_cli.call_count == 0


def test_prime_intellect_env_prepare_requires_prime_cli(monkeypatch):
    install_fake_verifiers(monkeypatch, load_environment=lambda env_id, **env_args: None)
    monkeypatch.setattr("marin.rl.environments.prime_intellect_env.shutil.which", lambda executable: None)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    with pytest.raises(RuntimeError, match="requires the 'prime' executable"):
        env.prepare()


def test_prime_intellect_env_sample_rejects_invalid_mode(monkeypatch, prime_cli, gpt2_tokenizer):
    verifier_env = FakeVerifierEnv(
        prompt_dataset(["0"], "train"),
        generate_result_factory=lambda *, inputs: single_turn_generate_outputs(inputs, gpt2_tokenizer),
    )
    load_calls = []

    def load_environment(env_id: str, **env_args):
        load_calls.append((env_id, dict(env_args)))
        return verifier_env

    install_fake_verifiers(monkeypatch, load_environment=load_environment)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="Unsupported mode"):
        env.sample(
            inference_ctx=levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
            mode="debug",
        )

    assert load_calls == []


def test_prime_intellect_env_sample_rejects_system_prompt(monkeypatch, prime_cli, gpt2_tokenizer):
    verifier_env = FakeVerifierEnv(
        prompt_dataset(["0"], "train"),
        generate_result_factory=lambda *, inputs: single_turn_generate_outputs(inputs, gpt2_tokenizer),
    )
    load_calls = []

    def load_environment(env_id: str, **env_args):
        load_calls.append((env_id, dict(env_args)))
        return verifier_env

    install_fake_verifiers(monkeypatch, load_environment=load_environment)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="does not support Marin-level system prompts"):
        env.sample(
            inference_ctx=levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
            system_prompt="You are helpful.",
        )

    assert load_calls == []


def test_prime_intellect_env_sample_rejects_non_chat_verifier_env(monkeypatch, prime_cli, gpt2_tokenizer):
    verifier_env = FakeVerifierEnv(
        prompt_dataset(["0"], "train"),
        message_type="completion",
        generate_result_factory=lambda *, inputs: single_turn_generate_outputs(inputs, gpt2_tokenizer),
    )
    install_fake_verifiers(monkeypatch, load_environment=lambda env_id, **env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="only supports chat-format verifier environments"):
        env.sample(
            inference_ctx=levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
        )


def test_prime_intellect_env_sample_rejects_tool_enabled_verifier_env(monkeypatch, prime_cli, gpt2_tokenizer):
    verifier_env = FakeVerifierEnv(
        prompt_dataset(["0"], "train"),
        tool_defs=[{"type": "function"}],
        generate_result_factory=lambda *, inputs: single_turn_generate_outputs(inputs, gpt2_tokenizer),
    )
    install_fake_verifiers(monkeypatch, load_environment=lambda env_id, **env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="does not support tool-enabled verifier environments"):
        env.sample(
            inference_ctx=levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
        )


def test_prime_intellect_env_sample_rejects_non_assistant_completion_turns(monkeypatch, prime_cli, gpt2_tokenizer):
    dataset = prompt_dataset(["0"], "train")

    def generate_result_factory(*, inputs):
        output = single_turn_generate_outputs(inputs, gpt2_tokenizer)
        output.completion[0] = [
            {"role": "assistant", "content": "resp-0-0"},
            {"role": "user", "content": "tool feedback"},
        ]
        return output

    verifier_env = FakeVerifierEnv(dataset, generate_result_factory=generate_result_factory)
    install_fake_verifiers(monkeypatch, load_environment=lambda env_id, **env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="does not support non-assistant turns in completions"):
        env.sample(
            inference_ctx=levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
        )


def test_prime_intellect_env_sample_rejects_multiple_response_objects(monkeypatch, prime_cli, gpt2_tokenizer):
    dataset = prompt_dataset(["0"], "train")

    def generate_result_factory(*, inputs):
        output = single_turn_generate_outputs(inputs, gpt2_tokenizer)
        response = output.state[0]["responses"][0]
        output.state[0] = {"responses": [response, response]}
        return output

    verifier_env = FakeVerifierEnv(dataset, generate_result_factory=generate_result_factory)
    install_fake_verifiers(monkeypatch, load_environment=lambda env_id, **env_args: verifier_env)
    env = PrimeIntellectEnv(env_id="primeintellect/gsm8k")

    env.prepare()
    with pytest.raises(ValueError, match="requires exactly one response object per rollout"):
        env.sample(
            inference_ctx=levanter_inference_ctx(gpt2_tokenizer),
            n_examples=1,
            n_generations=1,
            temperature=1.0,
            prng_key=None,
        )
