# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic rendering helpers for agent-trace transcripts."""

import json
from collections.abc import Mapping, Sequence
from typing import Any
from xml.sax.saxutils import quoteattr

from marin.transform.agent_traces.common import (
    TraceTranscriptConfig,
    TraceTranscriptMetadataValue,
    TraceTranscriptRow,
)

TRUNCATED_MARKER_TEMPLATE = '\n<truncated original_chars="{original_chars}"/>\n'
TRACE_TAG = "trace"
TOOLS_TAG = "tools"
TASK_TAG = "task"
OUTCOME_TAG = "outcome"
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
TOOL_CALL_TAG = "tool_call"
TOOL_RESPONSE_TAG = "tool_response"
DOUBLE_NEWLINE = "\n\n"


def _serialize_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _render_block(tag: str, content: str, *, attrs: Mapping[str, str] | None = None) -> str:
    attr_string = ""
    if attrs:
        rendered_attrs = [f"{key}={quoteattr(value)}" for key, value in attrs.items() if value]
        if rendered_attrs:
            attr_string = " " + " ".join(rendered_attrs)
    return f"<{tag}{attr_string}>\n{content}\n</{tag}>"


def _render_tool_response_block(payload: Any, *, max_chars: int, attrs: Mapping[str, str] | None = None) -> str:
    return _render_block(TOOL_RESPONSE_TAG, render_tool_payload(payload, max_chars=max_chars), attrs=attrs)


def _normalize_tool_call_payload(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    tool_name = tool_call.get("tool_name") or tool_call.get("name")
    if not tool_name:
        raise ValueError("tool call is missing a name.")
    arguments = tool_call.get("arguments")
    if arguments is None:
        arguments = tool_call.get("input", {})
    return {
        "name": tool_name,
        "arguments": arguments,
    }


def _observation_payload(observation: Mapping[str, Any]) -> Any:
    content = observation.get("content")
    output_summary = observation.get("output_summary")
    error = observation.get("error")

    if content not in (None, "") and output_summary in (None, "") and error in (None, ""):
        return content

    payload: dict[str, Any] = {}
    for key in ("content", "output_summary", "error"):
        value = observation.get(key)
        if value not in (None, "", [], {}):
            payload[key] = value

    if payload:
        return payload

    return {
        key: value for key, value in observation.items() if key != "source_call_id" and value not in (None, "", [], {})
    }


def _render_reasoning(reasoning_content: str) -> str:
    return f"{THINK_OPEN_TAG}\n{reasoning_content}\n{THINK_CLOSE_TAG}"


def _task_text(task: str | Mapping[str, Any] | None, *, max_chars: int) -> str:
    if task is None:
        return ""
    if isinstance(task, str):
        return truncate_large_payload(task, max_chars=max_chars)

    description = task.get("description")
    if isinstance(description, str) and description:
        return truncate_large_payload(description, max_chars=max_chars)

    return render_tool_payload(task, max_chars=max_chars)


def _outcome_status(outcome: Mapping[str, Any] | None) -> str | None:
    if not outcome:
        return None
    success = outcome.get("success")
    if success is True:
        return "success"
    if success is False:
        return "failure"
    terminal_state = outcome.get("terminal_state")
    if isinstance(terminal_state, str) and terminal_state:
        return terminal_state
    return None


def _tool_count(steps: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(step.get("tool_calls") or []) for step in steps)


def truncate_large_payload(text: str, *, max_chars: int) -> str:
    """Truncate a payload deterministically while preserving both ends."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if len(text) <= max_chars:
        return text

    marker = TRUNCATED_MARKER_TEMPLATE.format(original_chars=len(text))
    if len(marker) >= max_chars:
        return text[:max_chars]

    head_chars = (max_chars - len(marker)) // 2
    tail_chars = max_chars - len(marker) - head_chars
    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"


def render_tool_payload(payload: Any, *, max_chars: int) -> str:
    """Render dict/list/string tool payloads into a deterministic text representation."""
    return truncate_large_payload(_serialize_payload(payload), max_chars=max_chars)


def render_tools_block(tool_definitions: Sequence[Any], *, max_chars: int) -> str:
    """Render the shared `<tools>` block used by transcript datasets."""
    if not tool_definitions:
        return ""
    rendered_tools = [render_tool_payload(tool_definition, max_chars=max_chars) for tool_definition in tool_definitions]
    return _render_block(TOOLS_TAG, "\n".join(rendered_tools))


def render_trace_header(*, source: str, trace_id: str) -> str:
    """Render the opening `<trace>` element with stable provenance attributes."""
    if not source:
        raise ValueError("source must be non-empty.")
    if not trace_id:
        raise ValueError("trace_id must be non-empty.")
    return f"<{TRACE_TAG} source={quoteattr(source)} trace_id={quoteattr(trace_id)}>"


def render_step_block(step: Mapping[str, Any], *, max_tool_output_chars: int) -> str:
    """Render one normalized trace step into transcript blocks."""
    role = str(step.get("role", "")).strip()
    if not role:
        raise ValueError("step role is required.")

    rendered_sections: list[str] = []

    content = step.get("content")
    reasoning_content = step.get("reasoning_content")
    if role == "tool":
        rendered_sections.append(
            _render_tool_response_block(
                content,
                max_chars=max_tool_output_chars,
                attrs={
                    "name": str(step.get("name", "") or ""),
                    "id": str(step.get("tool_call_id", "") or ""),
                },
            )
        )
    else:
        role_parts: list[str] = []
        if isinstance(reasoning_content, str) and reasoning_content:
            role_parts.append(_render_reasoning(reasoning_content))
        if isinstance(content, str) and content:
            role_parts.append(content)
        elif content not in (None, "") and not isinstance(content, str):
            role_parts.append(render_tool_payload(content, max_chars=max_tool_output_chars))

        if role_parts:
            rendered_sections.append(_render_block(role, "\n".join(role_parts)))

    for tool_call in step.get("tool_calls") or []:
        rendered_sections.append(
            _render_block(
                TOOL_CALL_TAG,
                render_tool_payload(_normalize_tool_call_payload(tool_call), max_chars=max_tool_output_chars),
            )
        )

    for observation in step.get("observations") or []:
        attrs: dict[str, str] = {}
        source_call_id = observation.get("source_call_id")
        if isinstance(source_call_id, str) and source_call_id:
            attrs["id"] = source_call_id
        rendered_sections.append(
            _render_tool_response_block(
                _observation_payload(observation),
                max_chars=max_tool_output_chars,
                attrs=attrs or None,
            )
        )

    return DOUBLE_NEWLINE.join(section for section in rendered_sections if section)


def render_trace_transcript_row(
    *,
    trace_id: str,
    source: str,
    config: TraceTranscriptConfig,
    task: str | Mapping[str, Any] | None,
    tool_definitions: Sequence[Any],
    steps: Sequence[Mapping[str, Any]],
    outcome: Mapping[str, Any] | None,
    metadata: Mapping[str, TraceTranscriptMetadataValue] | None = None,
) -> TraceTranscriptRow:
    """Render a normalized trace into the canonical text-plus-metadata row."""
    sections: list[str] = [render_trace_header(source=source, trace_id=trace_id)]

    task_text = _task_text(task, max_chars=config.max_tool_output_chars)
    if task_text:
        sections.append(_render_block(TASK_TAG, task_text))

    tools_block = render_tools_block(tool_definitions, max_chars=config.max_tool_output_chars)
    if tools_block:
        sections.append(tools_block)

    for step in steps:
        rendered_step = render_step_block(step, max_tool_output_chars=config.max_tool_output_chars)
        if rendered_step:
            sections.append(rendered_step)

    if outcome:
        sections.append(_render_block(OUTCOME_TAG, render_tool_payload(outcome, max_chars=config.max_tool_output_chars)))

    sections.append(f"</{TRACE_TAG}>")
    transcript_text = DOUBLE_NEWLINE.join(sections)

    row_metadata = dict(metadata or {})
    row_metadata.setdefault("trace_source", source)
    row_metadata.setdefault("source_id", trace_id)
    row_metadata.setdefault("tool_count", _tool_count(steps))
    row_metadata.setdefault("step_count", len(steps))

    outcome_status = _outcome_status(outcome)
    if outcome_status is not None:
        row_metadata.setdefault("outcome_status", outcome_status)

    return TraceTranscriptRow(
        id=trace_id,
        source=source,
        text=transcript_text,
        metadata=row_metadata,
    )
