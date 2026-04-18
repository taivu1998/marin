# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert Pi session JSONL files into branch-linearized Dolma transcript rows."""

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

import draccus
from marin.execution.executor import InputName, THIS_OUTPUT_PATH
from marin.transform.agent_traces.common import (
    TraceBranchPolicy,
    TraceSourceFormat,
    TraceTranscriptConfig,
    TraceTranscriptMetadataValue,
)
from marin.transform.agent_traces.transcript_rendering import render_trace_transcript_row
from rigging.filesystem import open_url
from zephyr import Dataset, ZephyrContext

PI_MONO_SOURCE = "badlogicgames/pi-mono"
PI_MONO_HARNESS = "pi"

SESSION_ENTRY_TYPE = "session"
MESSAGE_ENTRY_TYPE = "message"
MODEL_CHANGE_ENTRY_TYPE = "model_change"
THINKING_LEVEL_CHANGE_ENTRY_TYPE = "thinking_level_change"
COMPACTION_ENTRY_TYPE = "compaction"
BRANCH_SUMMARY_ENTRY_TYPE = "branch_summary"
CUSTOM_ENTRY_TYPE = "custom"
CUSTOM_MESSAGE_ENTRY_TYPE = "custom_message"
LABEL_ENTRY_TYPE = "label"
SESSION_INFO_ENTRY_TYPE = "session_info"

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL_RESULT = "toolResult"
ROLE_BASH_EXECUTION = "bashExecution"
ROLE_CUSTOM = "custom"

TEXT_BLOCK_TYPE = "text"
IMAGE_BLOCK_TYPE = "image"
THINKING_BLOCK_TYPE = "thinking"
TOOL_CALL_BLOCK_TYPE = "toolCall"

NEWLINE = "\n"
IMAGE_PLACEHOLDER_TEMPLATE = "[image omitted mime_type={mime_type}]"
UNKNOWN_BLOCK_PLACEHOLDER_TEMPLATE = "[content omitted type={block_type}]"
CUSTOM_TYPE_PREFIX_TEMPLATE = "[custom_type={custom_type}]"
ERROR_PREFIX = "[error]"
NO_OUTPUT_PLACEHOLDER = "(no output)"
CAMEL_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


@dataclass(frozen=True)
class ExtractedContent:
    """Text extracted from content blocks plus filtering counters."""

    text: str
    image_count: int = 0
    omitted_block_count: int = 0


@dataclass(frozen=True)
class PiSessionToDolmaConfig:
    """Configuration for converting Pi sessions into transcript rows."""

    input_path: str | InputName
    output_path: str = THIS_OUTPUT_PATH
    input_glob: str = "20*.jsonl"
    max_tool_output_chars: int = 4000

    def transcript_config(self) -> TraceTranscriptConfig:
        """Build the shared transcript-rendering config used by this source."""
        return TraceTranscriptConfig(
            input_path=str(self.input_path),
            output_path=self.output_path,
            source_format=TraceSourceFormat.PI_SESSION,
            max_tool_output_chars=self.max_tool_output_chars,
            branch_policy=TraceBranchPolicy.ROOT_TO_LEAF,
        )


def _normalized_role_tag(role: str) -> str:
    return CAMEL_CASE_BOUNDARY.sub("_", role).replace("-", "_").lower()


def _path_basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _branch_trace_id(session_id: str, leaf_id: str) -> str:
    return f"{session_id}:{leaf_id}"


def _scalar_metadata_value(value: Any) -> TraceTranscriptMetadataValue | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str | int | float):
        return value
    return None


def _require_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _json_payload_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _with_prefix(text: str, prefix: str | None) -> str:
    if not prefix:
        return text
    if not text:
        return prefix
    return f"{prefix}\n{text}"


def _extract_content_text(content: Any) -> ExtractedContent:
    if content is None:
        return ExtractedContent("")
    if isinstance(content, str):
        return ExtractedContent(content)
    if not isinstance(content, Sequence) or isinstance(content, str):
        return ExtractedContent(_json_payload_text(content))

    parts: list[str] = []
    image_count = 0
    omitted_block_count = 0

    for block in content:
        if not isinstance(block, Mapping):
            parts.append(UNKNOWN_BLOCK_PLACEHOLDER_TEMPLATE.format(block_type=type(block).__name__))
            omitted_block_count += 1
            continue

        block_type = str(block.get("type", "")).strip()
        if block_type == TEXT_BLOCK_TYPE:
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
            continue

        if block_type == IMAGE_BLOCK_TYPE:
            image_count += 1
            mime_type = block.get("mimeType")
            placeholder = IMAGE_PLACEHOLDER_TEMPLATE.format(
                mime_type=mime_type if isinstance(mime_type, str) and mime_type else "unknown"
            )
            parts.append(placeholder)
            continue

        parts.append(UNKNOWN_BLOCK_PLACEHOLDER_TEMPLATE.format(block_type=block_type or "unknown"))
        omitted_block_count += 1

    return ExtractedContent(
        text="\n".join(part for part in parts if part),
        image_count=image_count,
        omitted_block_count=omitted_block_count,
    )


def _extract_assistant_step(message: Mapping[str, Any]) -> tuple[dict[str, Any] | None, int, int]:
    content = message.get("content")
    if isinstance(content, str):
        return (
            {
                "role": ROLE_ASSISTANT,
                "content": content,
                "tool_calls": [],
                "observations": [],
            },
            0,
            0,
        )
    if not isinstance(content, Sequence) or isinstance(content, str):
        payload = _json_payload_text(content)
        if not payload:
            return None, 0, 0
        return (
            {
                "role": ROLE_ASSISTANT,
                "content": payload,
                "tool_calls": [],
                "observations": [],
            },
            0,
            0,
        )

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    image_count = 0
    omitted_block_count = 0

    for block in content:
        if not isinstance(block, Mapping):
            omitted_block_count += 1
            text_parts.append(UNKNOWN_BLOCK_PLACEHOLDER_TEMPLATE.format(block_type=type(block).__name__))
            continue

        block_type = str(block.get("type", "")).strip()
        if block_type == TEXT_BLOCK_TYPE:
            text = block.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
            continue

        if block_type == THINKING_BLOCK_TYPE:
            thinking = block.get("thinking")
            if isinstance(thinking, str) and thinking:
                reasoning_parts.append(thinking)
            continue

        if block_type == TOOL_CALL_BLOCK_TYPE:
            name = _require_string(block.get("name"), name="assistant toolCall name")
            tool_calls.append(
                {
                    "name": name,
                    "arguments": block.get("arguments", {}),
                }
            )
            continue

        if block_type == IMAGE_BLOCK_TYPE:
            image_count += 1
            mime_type = block.get("mimeType")
            text_parts.append(
                IMAGE_PLACEHOLDER_TEMPLATE.format(
                    mime_type=mime_type if isinstance(mime_type, str) and mime_type else "unknown"
                )
            )
            continue

        omitted_block_count += 1
        text_parts.append(UNKNOWN_BLOCK_PLACEHOLDER_TEMPLATE.format(block_type=block_type or "unknown"))

    if not text_parts and not reasoning_parts and not tool_calls:
        return None, image_count, omitted_block_count

    step: dict[str, Any] = {
        "role": ROLE_ASSISTANT,
        "content": "\n".join(text_parts),
        "tool_calls": tool_calls,
        "observations": [],
    }
    if reasoning_parts:
        step["reasoning_content"] = "\n".join(reasoning_parts)
    return step, image_count, omitted_block_count


def _extract_text_step(*, role: str, content: Any, prefix: str | None = None) -> tuple[dict[str, Any] | None, int, int]:
    extracted = _extract_content_text(content)
    text = _with_prefix(extracted.text, prefix)
    if not text:
        return None, extracted.image_count, extracted.omitted_block_count
    return (
        {
            "role": role,
            "content": text,
            "tool_calls": [],
            "observations": [],
        },
        extracted.image_count,
        extracted.omitted_block_count,
    )


def _extract_tool_result_step(message: Mapping[str, Any], *, entry_id: str) -> tuple[dict[str, Any], int, int, bool]:
    tool_name = _require_string(message.get("toolName"), name="toolResult toolName")
    tool_call_id = message.get("toolCallId")
    extracted = _extract_content_text(message.get("content"))
    content_text = extracted.text or NO_OUTPUT_PLACEHOLDER
    is_error = bool(message.get("isError"))
    if is_error:
        content_text = _with_prefix(content_text, ERROR_PREFIX)

    return (
        {
            "role": "tool",
            "name": tool_name,
            "tool_call_id": tool_call_id if isinstance(tool_call_id, str) and tool_call_id else entry_id,
            "content": content_text,
        },
        extracted.image_count,
        extracted.omitted_block_count,
        is_error,
    )


def _extract_bash_execution_step(message: Mapping[str, Any], *, entry_id: str) -> tuple[dict[str, Any], int, int, bool]:
    command = _require_string(message.get("command"), name="bashExecution command")
    output = message.get("output")
    output_text = output if isinstance(output, str) and output else _json_payload_text(output) or NO_OUTPUT_PLACEHOLDER

    exit_code = message.get("exitCode")
    cancelled = bool(message.get("cancelled"))
    truncated = bool(message.get("truncated"))
    is_error = isinstance(exit_code, int) and exit_code != 0

    observation_payload: dict[str, Any] = {
        "output": output_text,
        "cancelled": cancelled,
        "truncated": truncated,
    }
    if isinstance(exit_code, int):
        observation_payload["exit_code"] = exit_code

    return (
        {
            "role": ROLE_ASSISTANT,
            "content": "",
            "tool_calls": [{"name": "bash", "arguments": {"command": command}}],
            "observations": [{"source_call_id": entry_id, "content": observation_payload}],
        },
        0,
        0,
        is_error,
    )


def _message_entry_step(
    entry: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, int, int, bool]:
    message = entry.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("Pi message entries must contain a message mapping.")

    role = _require_string(message.get("role"), name="Pi message role")
    entry_id = _require_string(entry.get("id"), name="Pi entry id")

    if role == ROLE_ASSISTANT:
        step, image_count, omitted_block_count = _extract_assistant_step(message)
        return step, image_count, omitted_block_count, False

    if role == ROLE_USER:
        step, image_count, omitted_block_count = _extract_text_step(role=ROLE_USER, content=message.get("content"))
        return step, image_count, omitted_block_count, False

    if role == ROLE_TOOL_RESULT:
        return _extract_tool_result_step(message, entry_id=entry_id)

    if role == ROLE_BASH_EXECUTION:
        if message.get("excludeFromContext") is True:
            return None, 0, 0, False
        return _extract_bash_execution_step(message, entry_id=entry_id)

    if role == ROLE_CUSTOM:
        custom_type = message.get("customType")
        prefix = None
        if isinstance(custom_type, str) and custom_type:
            prefix = CUSTOM_TYPE_PREFIX_TEMPLATE.format(custom_type=custom_type)
        step, image_count, omitted_block_count = _extract_text_step(
            role=ROLE_CUSTOM,
            content=message.get("content"),
            prefix=prefix,
        )
        return step, image_count, omitted_block_count, False

    step, image_count, omitted_block_count = _extract_text_step(
        role=_normalized_role_tag(role),
        content=message.get("content"),
    )
    return step, image_count, omitted_block_count, False


def _split_session_trace_entries(
    traces: Sequence[Any],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    header: Mapping[str, Any] | None = None
    entries: list[Mapping[str, Any]] = []

    for trace in traces:
        if not isinstance(trace, Mapping):
            raise ValueError("Pi session traces must be mappings.")
        entry_type = trace.get("type")
        if entry_type == SESSION_ENTRY_TYPE:
            if header is not None:
                raise ValueError("Pi session rows must contain exactly one session header.")
            header = trace
            continue
        entries.append(trace)

    if header is None:
        raise ValueError("Pi session rows must contain a session header.")
    return header, entries


def _append_leaf_paths(
    entry_id: str,
    *,
    children_by_parent: Mapping[str | None, list[str]],
    paths: list[list[str]],
    prefix: list[str],
) -> None:
    if entry_id in prefix:
        raise ValueError(f"Pi session tree contains a cycle involving entry {entry_id}.")

    current_path = [*prefix, entry_id]
    children = children_by_parent.get(entry_id, [])
    if not children:
        paths.append(current_path)
        return

    for child_id in children:
        _append_leaf_paths(child_id, children_by_parent=children_by_parent, paths=paths, prefix=current_path)


def _common_prefix_length(left: Sequence[str], right: Sequence[str]) -> int:
    common = 0
    for left_id, right_id in zip(left, right, strict=False):
        if left_id != right_id:
            break
        common += 1
    return common


def _shared_prefix_lengths(paths: Sequence[Sequence[str]]) -> dict[str, int]:
    if len(paths) <= 1:
        return {path[-1]: 0 for path in paths}

    shared_prefix_by_leaf: dict[str, int] = {}
    for index, path in enumerate(paths):
        longest_shared_prefix = 0
        for other_index, other_path in enumerate(paths):
            if index == other_index:
                continue
            longest_shared_prefix = max(longest_shared_prefix, _common_prefix_length(path, other_path))
        shared_prefix_by_leaf[path[-1]] = longest_shared_prefix
    return shared_prefix_by_leaf


def _session_paths(entries: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], list[list[str]]]:
    entries_by_id: dict[str, Mapping[str, Any]] = {}
    children_by_parent: dict[str | None, list[str]] = defaultdict(list)
    ordered_entry_ids: list[str] = []

    for entry in entries:
        entry_id = _require_string(entry.get("id"), name="Pi session entry id")
        if entry_id in entries_by_id:
            raise ValueError(f"Duplicate Pi session entry id: {entry_id}")
        parent_id = entry.get("parentId")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError(f"parentId for entry {entry_id} must be a string or null.")

        entries_by_id[entry_id] = entry
        ordered_entry_ids.append(entry_id)
        children_by_parent[parent_id].append(entry_id)

    roots = [
        entry_id
        for entry_id in ordered_entry_ids
        if (parent_id := entries_by_id[entry_id].get("parentId")) is None or parent_id not in entries_by_id
    ]
    if not roots:
        return entries_by_id, []

    paths: list[list[str]] = []
    for root_id in roots:
        _append_leaf_paths(root_id, children_by_parent=children_by_parent, paths=paths, prefix=[])
    return entries_by_id, paths


def _path_task_statement(*, session_name: str | None, first_user_message: str | None) -> str | None:
    if session_name:
        return session_name
    if first_user_message:
        return first_user_message
    return None


def _branch_summary_step(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    summary = entry.get("summary")
    if not isinstance(summary, str) or not summary:
        return None
    return {
        "role": BRANCH_SUMMARY_ENTRY_TYPE,
        "content": summary,
        "tool_calls": [],
        "observations": [],
    }


def _custom_message_step(entry: Mapping[str, Any]) -> tuple[dict[str, Any] | None, int, int]:
    custom_type = entry.get("customType")
    prefix = None
    if isinstance(custom_type, str) and custom_type:
        prefix = CUSTOM_TYPE_PREFIX_TEMPLATE.format(custom_type=custom_type)
    return _extract_text_step(role=ROLE_CUSTOM, content=entry.get("content"), prefix=prefix)


def _path_to_dolma_row(
    *,
    session_id: str,
    file_path: str,
    path_entries: Sequence[Mapping[str, Any]],
    branch_count: int,
    shared_prefix_length: int,
    session_version: TraceTranscriptMetadataValue | None,
    config: PiSessionToDolmaConfig,
) -> dict[str, str | dict[str, TraceTranscriptMetadataValue]] | None:
    session_name: str | None = None
    first_user_message: str | None = None
    last_provider: str | None = None
    last_model_id: str | None = None
    last_thinking_level: str | None = None
    filtered_event_count = 0
    image_placeholder_count = 0
    omitted_content_block_count = 0
    branch_summary_count = 0
    custom_message_count = 0
    tool_error_count = 0
    steps: list[Mapping[str, Any]] = []

    for entry in path_entries:
        entry_type = _require_string(entry.get("type"), name="Pi session entry type")

        if entry_type == MESSAGE_ENTRY_TYPE:
            step, image_count, omitted_block_count, is_error = _message_entry_step(entry)
            image_placeholder_count += image_count
            omitted_content_block_count += omitted_block_count
            if is_error:
                tool_error_count += 1
            if step is None:
                filtered_event_count += 1
                continue
            steps.append(step)

            if step["role"] == ROLE_USER and first_user_message is None:
                content = step.get("content")
                if isinstance(content, str) and content:
                    first_user_message = content

            message = entry["message"]
            if isinstance(message, Mapping):
                provider = message.get("provider")
                if isinstance(provider, str) and provider:
                    last_provider = provider
                model = message.get("model")
                if isinstance(model, str) and model:
                    last_model_id = model
                if step["role"] == ROLE_CUSTOM:
                    custom_message_count += 1
            continue

        if entry_type == BRANCH_SUMMARY_ENTRY_TYPE:
            step = _branch_summary_step(entry)
            if step is None:
                filtered_event_count += 1
                continue
            branch_summary_count += 1
            steps.append(step)
            continue

        if entry_type == CUSTOM_MESSAGE_ENTRY_TYPE:
            step, image_count, omitted_block_count = _custom_message_step(entry)
            image_placeholder_count += image_count
            omitted_content_block_count += omitted_block_count
            if step is None:
                filtered_event_count += 1
                continue
            custom_message_count += 1
            steps.append(step)
            continue

        if entry_type == SESSION_INFO_ENTRY_TYPE:
            name = entry.get("name")
            if isinstance(name, str) and name:
                session_name = name
            filtered_event_count += 1
            continue

        if entry_type == MODEL_CHANGE_ENTRY_TYPE:
            provider = entry.get("provider")
            if isinstance(provider, str) and provider:
                last_provider = provider
            model_id = entry.get("modelId")
            if isinstance(model_id, str) and model_id:
                last_model_id = model_id
            filtered_event_count += 1
            continue

        if entry_type == THINKING_LEVEL_CHANGE_ENTRY_TYPE:
            thinking_level = entry.get("thinkingLevel")
            if isinstance(thinking_level, str) and thinking_level:
                last_thinking_level = thinking_level
            filtered_event_count += 1
            continue

        if entry_type in {COMPACTION_ENTRY_TYPE, CUSTOM_ENTRY_TYPE, LABEL_ENTRY_TYPE}:
            filtered_event_count += 1
            continue

        filtered_event_count += 1

    if not steps:
        return None

    leaf_id = _require_string(path_entries[-1].get("id"), name="Pi branch leaf id")
    metadata: dict[str, TraceTranscriptMetadataValue] = {
        "harness": PI_MONO_HARNESS,
        "session_id": session_id,
        "file_path": _path_basename(file_path),
        "leaf_id": leaf_id,
        "branch_depth": len(path_entries),
        "shared_prefix_length": shared_prefix_length,
        "branch_count": branch_count,
        "filtered_event_count": filtered_event_count,
    }
    if session_version is not None:
        metadata["session_version"] = session_version
    if session_name:
        metadata["session_name"] = session_name
    if last_provider:
        metadata["provider"] = last_provider
    if last_model_id:
        metadata["model_id"] = last_model_id
    if last_thinking_level:
        metadata["thinking_level"] = last_thinking_level
    if image_placeholder_count:
        metadata["image_placeholder_count"] = image_placeholder_count
    if omitted_content_block_count:
        metadata["omitted_content_block_count"] = omitted_content_block_count
    if branch_summary_count:
        metadata["branch_summary_count"] = branch_summary_count
    if custom_message_count:
        metadata["custom_message_count"] = custom_message_count
    if tool_error_count:
        metadata["tool_error_count"] = tool_error_count

    row = render_trace_transcript_row(
        trace_id=_branch_trace_id(session_id, leaf_id),
        source=PI_MONO_SOURCE,
        config=config.transcript_config(),
        task=_path_task_statement(session_name=session_name, first_user_message=first_user_message),
        tool_definitions=[],
        steps=steps,
        outcome=None,
        metadata=metadata,
    )
    return row.to_dict()


def _session_entries_to_dolma_rows(
    *,
    header: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    file_path: str,
    config: PiSessionToDolmaConfig,
) -> list[dict[str, str | dict[str, TraceTranscriptMetadataValue]]]:
    session_id = _require_string(header.get("id"), name="Pi session header id")
    session_version = _scalar_metadata_value(header.get("version"))
    entries_by_id, paths = _session_paths(entries)
    if not paths:
        return []

    shared_prefix_by_leaf = _shared_prefix_lengths(paths)
    rows: list[dict[str, str | dict[str, TraceTranscriptMetadataValue]]] = []
    for path in paths:
        path_entries = [entries_by_id[entry_id] for entry_id in path]
        row = _path_to_dolma_row(
            session_id=session_id,
            file_path=file_path,
            path_entries=path_entries,
            branch_count=len(paths),
            shared_prefix_length=shared_prefix_by_leaf[path[-1]],
            session_version=session_version,
            config=config,
        )
        if row is not None:
            rows.append(row)
    return rows


def pi_session_record_to_dolma_rows(
    record: Mapping[str, Any], config: PiSessionToDolmaConfig
) -> list[dict[str, str | dict[str, TraceTranscriptMetadataValue]]]:
    """Convert one dataset-style Pi session row into branch transcript rows."""
    traces = record.get("traces")
    if not isinstance(traces, Sequence) or isinstance(traces, str):
        raise ValueError("Pi dataset rows must contain a non-empty traces sequence.")

    file_path = _require_string(record.get("file_path"), name="Pi file_path")
    header, entries = _split_session_trace_entries(traces)

    session_id = record.get("session_id")
    if session_id is not None and session_id != header.get("id"):
        raise ValueError("Pi dataset row session_id must match the session header id.")

    return _session_entries_to_dolma_rows(
        header=header,
        entries=entries,
        file_path=file_path,
        config=config,
    )


def pi_session_file_to_dolma_rows(
    file_path: str, config: PiSessionToDolmaConfig
) -> Iterator[dict[str, str | dict[str, TraceTranscriptMetadataValue]]]:
    """Load one raw Pi session file and yield one Dolma row per root-to-leaf branch."""
    traces: list[Mapping[str, Any]] = []
    with open_url(file_path, "rt", compression="infer") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            trace = json.loads(stripped)
            if not isinstance(trace, Mapping):
                raise ValueError("Pi session JSONL lines must decode into mappings.")
            traces.append(trace)

    header, entries = _split_session_trace_entries(traces)
    yield from _session_entries_to_dolma_rows(
        header=header,
        entries=entries,
        file_path=_path_basename(file_path),
        config=config,
    )


def convert_pi_session_to_dolma(config: PiSessionToDolmaConfig) -> None:
    """Transform raw Pi session files into Dolma-format transcript rows."""
    pipeline = (
        Dataset.from_files(f"{config.input_path}/{config.input_glob}", empty_glob_ok=False)
        .flat_map(lambda path: pi_session_file_to_dolma_rows(path, config))
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(name="pi-session-to-dolma")
    ctx.execute(pipeline)


if __name__ == "__main__":
    draccus.wrap(convert_pi_session_to_dolma)()
