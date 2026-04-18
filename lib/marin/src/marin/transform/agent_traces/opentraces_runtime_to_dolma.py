# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert OpenTraces runtime JSONL files into Dolma-style transcript rows."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import draccus
from marin.execution.executor import InputName, THIS_OUTPUT_PATH
from marin.transform.agent_traces.common import TraceSourceFormat, TraceTranscriptConfig, TraceTranscriptMetadataValue
from marin.transform.agent_traces.transcript_rendering import render_trace_transcript_row
from zephyr import Dataset, ZephyrContext, load_jsonl

OPENTRACES_RUNTIME_SOURCE = "OpenTraces/opentraces-runtime"


@dataclass(frozen=True)
class OpenTracesRuntimeToDolmaConfig:
    """Configuration for converting OpenTraces runtime traces into transcript rows."""

    input_path: str | InputName
    output_path: str = THIS_OUTPUT_PATH
    input_glob: str = "data/traces_*.jsonl"
    max_tool_output_chars: int = 4000

    def transcript_config(self) -> TraceTranscriptConfig:
        """Build the shared transcript-rendering config used by this source."""
        return TraceTranscriptConfig(
            input_path=str(self.input_path),
            output_path=self.output_path,
            source_format=TraceSourceFormat.OPENTRACES_RUNTIME,
            max_tool_output_chars=self.max_tool_output_chars,
        )


def _scalar_metadata_value(value: Any) -> TraceTranscriptMetadataValue | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str | int | float):
        return value
    return None


def _selected_metadata(record: Mapping[str, Any]) -> dict[str, TraceTranscriptMetadataValue]:
    metadata: dict[str, TraceTranscriptMetadataValue] = {}

    for key in ("session_id", "schema_version", "execution_context"):
        value = _scalar_metadata_value(record.get(key))
        if value is not None:
            metadata[key] = value

    agent = record.get("agent")
    if isinstance(agent, Mapping):
        for source_key, metadata_key in (
            ("name", "agent_name"),
            ("version", "agent_version"),
            ("model", "agent_model"),
        ):
            value = _scalar_metadata_value(agent.get(source_key))
            if value is not None:
                metadata[metadata_key] = value

    task = record.get("task")
    if isinstance(task, Mapping):
        for source_key, metadata_key in (
            ("source", "task_source"),
            ("repository", "task_repository"),
            ("base_commit", "task_base_commit"),
        ):
            value = _scalar_metadata_value(task.get(source_key))
            if value is not None:
                metadata[metadata_key] = value

    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        for key in ("total_input_tokens", "total_output_tokens", "total_duration_s", "estimated_cost_usd"):
            value = _scalar_metadata_value(metrics.get(key))
            if value is not None:
                metadata[key] = value

    outcome = record.get("outcome")
    if isinstance(outcome, Mapping):
        value = _scalar_metadata_value(outcome.get("success"))
        if value is not None:
            metadata["success"] = value

    source_metadata = record.get("metadata")
    if isinstance(source_metadata, Mapping):
        for key, value in source_metadata.items():
            normalized_value = _scalar_metadata_value(value)
            if normalized_value is not None and key not in metadata:
                metadata[key] = normalized_value

    return metadata


def _tool_definitions(record: Mapping[str, Any]) -> list[Any]:
    tool_definitions = record.get("tool_definitions")
    if not isinstance(tool_definitions, Sequence) or isinstance(tool_definitions, str):
        return []
    return list(tool_definitions)


def _steps(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = record.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, str):
        return []
    normalized_steps = [step for step in steps if isinstance(step, Mapping)]
    return normalized_steps


def opentraces_runtime_record_to_dolma(
    record: Mapping[str, Any], config: OpenTracesRuntimeToDolmaConfig
) -> dict[str, str | dict[str, TraceTranscriptMetadataValue]] | None:
    """Convert one OpenTraces runtime record into the canonical transcript row."""
    trace_id = record.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("OpenTraces runtime records must have a non-empty trace_id.")

    steps = _steps(record)
    if not steps:
        return None

    task = record.get("task")
    normalized_task = task if isinstance(task, str | Mapping) else None

    outcome = record.get("outcome")
    normalized_outcome = outcome if isinstance(outcome, Mapping) else None

    row = render_trace_transcript_row(
        trace_id=trace_id,
        source=OPENTRACES_RUNTIME_SOURCE,
        config=config.transcript_config(),
        task=normalized_task,
        tool_definitions=_tool_definitions(record),
        steps=steps,
        outcome=normalized_outcome,
        metadata=_selected_metadata(record),
    )
    return row.to_dict()


def convert_opentraces_runtime_to_dolma(config: OpenTracesRuntimeToDolmaConfig) -> None:
    """Transform OpenTraces runtime files into Dolma-format transcript rows."""
    pipeline = (
        Dataset.from_files(f"{config.input_path}/{config.input_glob}", empty_glob_ok=False)
        .flat_map(load_jsonl)
        .map(lambda record: opentraces_runtime_record_to_dolma(record, config))
        .filter(lambda record: record is not None)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(name="opentraces-runtime-to-dolma")
    ctx.execute(pipeline)


if __name__ == "__main__":
    draccus.wrap(convert_opentraces_runtime_to_dolma)()
