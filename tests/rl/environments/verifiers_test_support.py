# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for Prime/verifiers-backed environment adapters."""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

from datasets import Dataset
from marin.rl.environments.inference_ctx import (
    LevanterInferenceContext,
    LevanterInferenceContextConfig,
    VLLMSamplingConfig,
    vLLMInferenceContext,
    vLLMInferenceContextConfig,
)
from marin.rl.environments.inference_ctx.vllm import InferenceMode
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletionTokenLogprob, Choice, ChoiceLogprobs
from openai.types.completion_usage import CompletionUsage


@dataclass
class DummyInferenceServer:
    """Minimal inference server for Levanter OpenAI client construction."""

    host: str = "localhost"
    port: int = 8000

    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def config(self):
        @dataclass
        class Config:
            model_name: str = "test-model"

        return Config()


class FakeVerifierEnv:
    """Verifier env test double with real Dataset behavior."""

    def __init__(
        self,
        dataset: Dataset,
        eval_dataset: Dataset | None = None,
        *,
        message_type: str = "chat",
        tool_defs: list[dict[str, object]] | None = None,
        generate_result_factory=None,
    ):
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.message_type = message_type
        self.tool_defs = tool_defs
        self.generate_result_factory = generate_result_factory
        self.generate_calls: list[dict[str, object]] = []

    def get_dataset(self, n: int = -1):
        if n > 0:
            return self.dataset.select(range(min(n, len(self.dataset))))
        return self.dataset

    def get_eval_dataset(self, n: int = -1):
        dataset = self.eval_dataset if self.eval_dataset is not None else self.dataset
        if n > 0:
            return dataset.select(range(min(n, len(dataset))))
        return dataset

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
        return self.generate_result_factory(inputs=inputs)


def prompt_dataset(example_ids: list[str], prefix: str) -> Dataset:
    """Build a simple chat-prompt dataset for Prime env tests."""
    return Dataset.from_dict(
        {
            "id": example_ids,
            "prompt": [[{"role": "user", "content": f"{prefix} prompt {example_id}"}] for example_id in example_ids],
            "answer": [""] * len(example_ids),
        }
    )


def reasoning_gym_dataset(rows: list[dict[str, object]]) -> Dataset:
    """Build a small HF dataset shaped like `vf.ReasoningGymEnv` rows."""
    return Dataset.from_list(rows)


def create_chat_completion(tokenizer, response_text: str, prompt_token_ids: list[int]) -> ChatCompletion:
    """Create a chat completion with attached token IDs for both inference backends."""
    response_token_ids = tokenizer.encode(response_text, add_special_tokens=False)
    logprobs_content = [
        ChatCompletionTokenLogprob(
            token=tokenizer.convert_ids_to_tokens(token_id),
            logprob=-0.1 * (index + 1),
            bytes=None,
            top_logprobs=[],
        )
        for index, token_id in enumerate(response_token_ids)
    ]

    completion = ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=response_text),
                logprobs=ChoiceLogprobs(content=logprobs_content),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=len(response_token_ids),
            prompt_tokens=len(prompt_token_ids),
            total_tokens=len(prompt_token_ids) + len(response_token_ids),
        ),
    )
    completion.choices[0].prompt_token_ids = prompt_token_ids
    completion.choices[0].response_token_ids = response_token_ids
    return completion


def single_turn_generate_outputs(
    inputs,
    tokenizer,
    *,
    reward_by_answer: dict[str, list[float]] | None = None,
    response_by_answer: dict[str, list[str]] | None = None,
    metric_name: str = "score",
    metric_scale: float = 10.0,
):
    """Create verifier-style rollout outputs from a dataset slice."""
    counts_by_answer: dict[str, int] = defaultdict(int)
    prompts = []
    completions = []
    states = []
    rewards = []
    metric_values = []

    for rollout_index, row in enumerate(inputs):
        answer_key = str(row["answer"])
        if not answer_key and "id" in row:
            answer_key = str(row["id"])
        generation_index = counts_by_answer[answer_key]
        counts_by_answer[answer_key] += 1

        response_texts = response_by_answer.get(answer_key) if response_by_answer else None
        if response_texts is None:
            response_text = f"resp-{answer_key}-{generation_index}"
        else:
            response_text = response_texts[generation_index]

        reward_values = reward_by_answer.get(answer_key) if reward_by_answer else None
        if reward_values is None:
            reward = float(rollout_index + 1)
        else:
            reward = float(reward_values[generation_index])

        prompt_messages = row.get("prompt")
        if prompt_messages is None:
            prompt_messages = [{"role": "user", "content": str(row["question"])}]

        prompt_token_ids = [100 + rollout_index, 200 + rollout_index]
        response = create_chat_completion(tokenizer, response_text, prompt_token_ids)

        prompts.append(prompt_messages)
        completions.append([{"role": "assistant", "content": response_text}])
        states.append({"responses": [response]})
        rewards.append(reward)
        metric_values.append(metric_scale * reward)

    return SimpleNamespace(
        prompt=prompts,
        completion=completions,
        state=states,
        reward=rewards,
        metrics={metric_name: metric_values},
    )


def install_fake_verifiers(
    monkeypatch,
    *,
    load_environment=None,
    reasoning_gym_env_cls=None,
) -> ModuleType:
    """Install a fake `verifiers` module into `sys.modules` for unit tests."""
    fake_verifiers = ModuleType("verifiers")
    if load_environment is not None:
        fake_verifiers.load_environment = load_environment
    if reasoning_gym_env_cls is not None:
        fake_verifiers.ReasoningGymEnv = reasoning_gym_env_cls
    monkeypatch.setitem(sys.modules, "verifiers", fake_verifiers)
    return fake_verifiers


def install_fake_reasoning_gym(monkeypatch) -> ModuleType:
    """Install a minimal `reasoning_gym` module so prepare() import checks pass."""
    fake_reasoning_gym = ModuleType("reasoning_gym")
    monkeypatch.setitem(sys.modules, "reasoning_gym", fake_reasoning_gym)
    return fake_reasoning_gym


def levanter_inference_ctx(gpt2_tokenizer) -> LevanterInferenceContext:
    """Create a Levanter inference context without starting a real server."""
    ctx = LevanterInferenceContext(
        LevanterInferenceContextConfig(
            inference_server_config=None,
            tokenizer=gpt2_tokenizer,
            stop_tokens=None,
            max_tokens=128,
            mesh=None,
            axis_mapping={},
        )
    )
    ctx._inference_server = DummyInferenceServer()
    return ctx


def vllm_inference_ctx(monkeypatch, gpt2_tokenizer) -> vLLMInferenceContext:
    """Create a vLLM inference context without importing the real engine."""
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_llm_engine",
        staticmethod(lambda _config: object()),
    )
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm.load_tokenizer",
        lambda _path: gpt2_tokenizer,
    )
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_renderer",
        staticmethod(lambda _model_name, _tokenizer: object()),
    )

    return vLLMInferenceContext(
        vLLMInferenceContextConfig(
            model_name="test-model",
            canonical_model_name="meta-llama/Llama-3.1-8B-Instruct",
            max_model_len=1024,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            sampling_params=VLLMSamplingConfig(),
            mode=InferenceMode.SYNC,
        )
    )
