# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small shared helpers for verifier-backed Marin environments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from openai.types.chat.chat_completion import Choice


def import_verifiers(import_hint: str) -> Any:
    """Import verifiers or raise a targeted installation error."""
    try:
        import verifiers as vf
    except ImportError as exc:
        raise ImportError(import_hint) from exc

    return vf


def freeze_cache_value(value: object) -> object:
    """Freeze a JSON-like structure so it can be used as a cache key."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, list | tuple):
        return tuple(freeze_cache_value(item) for item in value)
    if isinstance(value, Mapping):
        frozen_items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Verifier env args keys must be strings, got {type(key).__name__}")
            frozen_items.append((key, freeze_cache_value(item)))
        return tuple(sorted(frozen_items))

    raise TypeError(
        "Verifier env args must contain only JSON-like values, " f"got unsupported type {type(value).__name__}"
    )


def scalarize_metric(metric_name: str, values: object) -> float:
    """Convert verifier metrics into one scalar float."""
    if isinstance(values, (int, float, np.number, bool)):
        return float(values)

    if not isinstance(values, list):
        raise TypeError(f"Metric {metric_name!r} must be numeric or a list of numeric values")
    if not values:
        raise ValueError(f"Metric {metric_name!r} cannot be an empty list")

    try:
        return float(np.mean(np.asarray(values, dtype=np.float32)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Metric {metric_name!r} must contain only numeric values") from exc


def validate_generate_outputs(result: Any, expected_rollouts: int) -> None:
    """Validate the common verifier output shape Marin expects."""
    for field_name in ("prompt", "completion", "state", "reward"):
        field_value = getattr(result, field_name, None)
        if not isinstance(field_value, list):
            raise ValueError(f"Verifier env expected result.{field_name} to be a list")
        if len(field_value) != expected_rollouts:
            raise ValueError(f"Verifier env expected {expected_rollouts} {field_name} entries, got {len(field_value)}")

    metrics = getattr(result, "metrics", {})
    if metrics is None:
        return
    if not isinstance(metrics, Mapping):
        raise ValueError("Verifier env expected result.metrics to be a mapping")


def extract_single_turn_chat_rollout(
    *,
    rollout_index: int,
    prompt: object,
    completion: object,
    state: object,
) -> tuple[list[dict[str, str]], Choice, str]:
    """Extract the single-turn chat rollout structure Marin supports today."""
    if not isinstance(prompt, list):
        raise ValueError(
            "Verifier env expected chat prompts as a list of messages, "
            f"got {type(prompt).__name__} for rollout {rollout_index}"
        )

    prompt_messages: list[dict[str, str]] = []
    for message_index, message in enumerate(prompt):
        if not isinstance(message, Mapping):
            raise TypeError(
                f"Verifier env expected prompt messages to be mappings, got {type(message).__name__} "
                f"at rollout {rollout_index} message {message_index}"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            raise ValueError(f"Verifier env prompt message {message_index} is missing a string role")
        if not isinstance(content, str):
            raise ValueError(f"Verifier env prompt message {message_index} is missing a string content")
        prompt_messages.append({"role": role, "content": content})

    if not isinstance(completion, list):
        raise ValueError(
            "Verifier env expected chat completions as a list of messages, "
            f"got {type(completion).__name__} for rollout {rollout_index}"
        )

    completion_messages: list[dict[str, str]] = []
    for message in completion:
        if not isinstance(message, Mapping):
            raise TypeError(f"Verifier env expected completion messages to be mappings, got {type(message).__name__}")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            raise ValueError("Verifier env completion messages must have a string role")
        if not isinstance(content, str):
            raise ValueError("Verifier env completion messages must have a string content")
        completion_messages.append({"role": role, "content": content})

    if any(message["role"] != "assistant" for message in completion_messages):
        raise ValueError("Verifier env does not support non-assistant turns in completions")
    if len(completion_messages) != 1:
        raise ValueError("Verifier env requires exactly one assistant completion turn")

    assistant_content = completion_messages[0]["content"]

    if not isinstance(state, Mapping):
        raise TypeError(f"Verifier env expected rollout state to be a mapping, got {type(state).__name__}")

    responses = state.get("responses")
    if not isinstance(responses, list):
        raise ValueError("Verifier env expected state['responses'] to be a list")
    if len(responses) != 1:
        raise ValueError("Verifier env requires exactly one response object per rollout")

    response = responses[0]
    if not hasattr(response, "choices"):
        raise ValueError("Verifier env expected state['responses'] entries to be ChatCompletion-like")
    if len(response.choices) != 1:
        raise ValueError("Verifier env requires exactly one assistant choice per rollout")

    choice = response.choices[0]
    if choice.message.role != "assistant":
        raise ValueError("Verifier env only supports assistant response choices")
    if choice.message.content is None:
        raise ValueError("Verifier env requires assistant responses with text content")
    if choice.message.content != assistant_content:
        raise ValueError("Verifier env requires completion messages to match response choices")

    return prompt_messages, choice, assistant_content
