# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for transcript-style agent trace rendering."""

import json
from pathlib import Path

import pytest

from marin.transform.agent_traces.common import TraceSourceFormat, TraceTranscriptConfig
from marin.transform.agent_traces.transcript_rendering import (
    render_step_block,
    render_tool_payload,
    render_tools_block,
    render_trace_header,
    render_trace_transcript_row,
    truncate_large_payload,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "agent_traces"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


def test_opentraces_schema_fixture_matches_sample_shape() -> None:
    schema = _load_fixture("opentraces_runtime_schema.json")
    sample = _load_fixture("opentraces_runtime_sample.json")

    assert set(schema["top_level_fields"]).issubset(sample.keys())
    assert set(schema["step_fields"]).issubset(sample["steps"][0].keys())
    assert set(schema["tool_call_fields"]).issubset(sample["steps"][1]["tool_calls"][0].keys())
    assert set(schema["observation_fields"]).issubset(sample["steps"][1]["observations"][0].keys())


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("plain text", "plain text"),
        ([{"b": 2, "a": 1}], '[{"a": 1, "b": 2}]'),
        ({"b": 2, "a": 1}, '{"a": 1, "b": 2}'),
    ],
)
def test_render_tool_payload_handles_string_list_and_dict(payload: object, expected: str) -> None:
    assert render_tool_payload(payload, max_chars=200) == expected


def test_truncate_large_payload_is_deterministic() -> None:
    payload = "abcdefghijklmnopqrstuvwxyz0123456789" * 4

    truncated_once = truncate_large_payload(payload, max_chars=64)
    truncated_twice = truncate_large_payload(payload, max_chars=64)

    assert truncated_once == truncated_twice
    assert len(truncated_once) == 64
    assert '<truncated original_chars="144"/>' in truncated_once
    assert truncated_once.startswith(payload[:10])
    assert truncated_once.endswith(payload[-10:])


def test_render_tools_block_uses_tools_tags_and_stable_json() -> None:
    tool_block = render_tools_block(
        [
            {"name": "grep", "description": "Search files", "parameters": {"type": "object"}},
            '{"name": "pytest", "description": "Run tests"}',
        ],
        max_chars=400,
    )

    assert tool_block.startswith("<tools>")
    assert tool_block.endswith("</tools>")
    assert '{"description": "Search files", "name": "grep", "parameters": {"type": "object"}}' in tool_block
    assert '{"name": "pytest", "description": "Run tests"}' in tool_block


def test_render_step_block_renders_tool_calls_and_observations() -> None:
    sample = _load_fixture("opentraces_runtime_sample.json")

    rendered = render_step_block(sample["steps"][1], max_tool_output_chars=400)

    assert "<assistant>" in rendered
    assert "<think>" in rendered
    assert "<tool_call>" in rendered
    assert '{"arguments": {"path": "tests/test_parser.py -k empty_input"}, "name": "pytest"}' in rendered
    assert '<tool_response id="call-1">' in rendered
    assert "Targeted parser test fails on empty input." in rendered


def test_render_step_block_supports_tool_role_messages() -> None:
    rendered = render_step_block(
        {
            "role": "tool",
            "name": "grep",
            "tool_call_id": "call-9",
            "content": {"exit_code": 0, "stdout": "match.py:12:TODO"},
            "tool_calls": [],
            "observations": [],
        },
        max_tool_output_chars=400,
    )

    assert rendered == (
        '<tool_response name="grep" id="call-9">\n' '{"exit_code": 0, "stdout": "match.py:12:TODO"}\n' "</tool_response>"
    )


def test_render_trace_header_requires_stable_source_and_id() -> None:
    assert render_trace_header(source="OpenTraces/opentraces-runtime", trace_id="trace-1") == (
        '<trace source="OpenTraces/opentraces-runtime" trace_id="trace-1">'
    )


def test_render_trace_transcript_row_propagates_source_id_and_metadata() -> None:
    sample = _load_fixture("opentraces_runtime_sample.json")
    config = TraceTranscriptConfig(
        input_path="/tmp/raw",
        output_path="/tmp/documents",
        source_format=TraceSourceFormat.OPENTRACES_RUNTIME,
        max_tool_output_chars=240,
    )

    row = render_trace_transcript_row(
        trace_id=sample["trace_id"],
        source="OpenTraces/opentraces-runtime",
        config=config,
        task=sample["task"],
        tool_definitions=sample["tool_definitions"],
        steps=sample["steps"],
        outcome=sample["outcome"],
        metadata={"task_category": sample["metadata"]["task_category"]},
    )

    assert row.id == sample["trace_id"]
    assert row.source == "OpenTraces/opentraces-runtime"
    assert row.metadata["source_id"] == sample["trace_id"]
    assert row.metadata["trace_source"] == "OpenTraces/opentraces-runtime"
    assert row.metadata["task_category"] == "coding"
    assert row.metadata["tool_count"] == 1
    assert row.metadata["step_count"] == 3
    assert row.metadata["outcome_status"] == "success"
    assert row.to_dict()["id"] == sample["trace_id"]


def test_rendered_transcript_has_expected_formatting_invariants() -> None:
    sample = _load_fixture("opentraces_runtime_sample.json")
    config = TraceTranscriptConfig(
        input_path="/tmp/raw",
        output_path="/tmp/documents",
        source_format=TraceSourceFormat.OPENTRACES_RUNTIME,
        max_tool_output_chars=180,
    )

    row = render_trace_transcript_row(
        trace_id=sample["trace_id"],
        source="OpenTraces/opentraces-runtime",
        config=config,
        task=sample["task"],
        tool_definitions=sample["tool_definitions"],
        steps=sample["steps"],
        outcome=sample["outcome"],
    )

    assert row.text.startswith('<trace source="OpenTraces/opentraces-runtime" trace_id="trace-opentraces-001">')
    assert row.text.endswith("</trace>")
    assert "\n\n\n" not in row.text
    assert "<task>" in row.text
    assert "<tools>" in row.text
    assert "<user>" in row.text
    assert "<assistant>" in row.text
    assert "<tool_call>" in row.text
    assert '<tool_response id="call-1">' in row.text
    assert "<outcome>" in row.text
