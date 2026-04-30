# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for transcript-style agent trace rendering."""

import json
from pathlib import Path

import pytest
from marin.transform.agent_traces.common import TraceSourceFormat, TraceTranscriptConfig
from marin.transform.agent_traces.opentraces_runtime_to_dolma import (
    OPENTRACES_RUNTIME_SOURCE,
    OpenTracesRuntimeToDolmaConfig,
    convert_opentraces_runtime_to_dolma,
    opentraces_runtime_record_to_dolma,
)
from marin.transform.agent_traces.pi_session_to_dolma import (
    PI_MONO_SOURCE,
    PiSessionToDolmaConfig,
    convert_pi_session_to_dolma,
    pi_session_record_to_dolma_rows,
)
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


def test_pi_mono_schema_fixture_matches_sample_shape() -> None:
    schema = _load_fixture("pi_mono_schema.json")
    sample = _load_fixture("pi_mono_session_sample.json")

    assert set(schema["top_level_fields"]).issubset(sample.keys())
    assert set(schema["entry_fields"]).issubset(sample["traces"][1].keys())
    assert {entry["type"] for entry in sample["traces"]}.issuperset(schema["control_entry_types"])
    assert {
        entry["message"]["role"]
        for entry in sample["traces"]
        if entry["type"] == "message" and "message" in entry and "role" in entry["message"]
    }.issuperset(schema["message_roles"])


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


def test_opentraces_runtime_record_to_dolma_promotes_selected_metadata(validate_dolma_record) -> None:
    sample = _load_fixture("opentraces_runtime_sample.json")
    config = OpenTracesRuntimeToDolmaConfig(input_path="/tmp/raw", output_path="/tmp/documents")

    result = opentraces_runtime_record_to_dolma(sample, config)

    assert result is not None
    validate_dolma_record(result)
    assert result["id"] == sample["trace_id"]
    assert result["source"] == OPENTRACES_RUNTIME_SOURCE
    assert result["metadata"]["session_id"] == sample["session_id"]
    assert result["metadata"]["agent_model"] == sample["agent"]["model"]
    assert result["metadata"]["task_repository"] == sample["task"]["repository"]
    assert result["metadata"]["total_input_tokens"] == sample["metrics"]["total_input_tokens"]
    assert result["metadata"]["success"] is True
    assert result["metadata"]["task_category"] == sample["metadata"]["task_category"]
    assert result["metadata"]["language"] == sample["metadata"]["language"]


def test_opentraces_runtime_record_to_dolma_skips_empty_step_records() -> None:
    sample = _load_fixture("opentraces_runtime_sample.json")
    sample["steps"] = []
    config = OpenTracesRuntimeToDolmaConfig(input_path="/tmp/raw", output_path="/tmp/documents")

    assert opentraces_runtime_record_to_dolma(sample, config) is None


def test_convert_opentraces_runtime_to_dolma_end_to_end(tmp_path, read_all_jsonl_gz, validate_dolma_record) -> None:
    sample = _load_fixture("opentraces_runtime_sample.json")
    input_dir = tmp_path / "raw" / "data"
    output_dir = tmp_path / "documents"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "traces_test.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

    convert_opentraces_runtime_to_dolma(
        OpenTracesRuntimeToDolmaConfig(
            input_path=str(tmp_path / "raw"),
            output_path=str(output_dir),
            input_glob="data/*.jsonl",
            max_tool_output_chars=180,
        )
    )

    records = read_all_jsonl_gz(output_dir)

    assert len(records) == 1
    validate_dolma_record(records[0])
    assert records[0]["id"] == sample["trace_id"]
    assert records[0]["metadata"]["trace_source"] == OPENTRACES_RUNTIME_SOURCE
    assert "<tool_call>" in records[0]["text"]
    assert "<outcome>" in records[0]["text"]


def test_pi_session_record_to_dolma_rows_extracts_root_to_leaf_branches(validate_dolma_record) -> None:
    sample = _load_fixture("pi_mono_session_sample.json")
    config = PiSessionToDolmaConfig(input_path="/tmp/raw", output_path="/tmp/documents")

    rows = pi_session_record_to_dolma_rows(sample, config)

    assert len(rows) == 2
    for row in rows:
        validate_dolma_record(row)
        assert row["source"] == PI_MONO_SOURCE
        assert row["metadata"]["session_id"] == sample["session_id"]
        assert row["metadata"]["branch_count"] == 2
        assert row["metadata"]["shared_prefix_length"] == 4

    main_branch, alternate_branch = rows

    assert main_branch["id"] == "session-pi-001:compaction-leaf"
    assert main_branch["metadata"]["leaf_id"] == "compaction-leaf"
    assert main_branch["metadata"]["branch_depth"] == 8
    assert main_branch["metadata"]["filtered_event_count"] == 4
    assert main_branch["metadata"]["session_name"] == "Fix parser empty-input regression"
    assert main_branch["metadata"]["image_placeholder_count"] == 1
    assert "<branch_summary>" not in main_branch["text"]
    assert "<tool_call>" in main_branch["text"]
    assert "[image omitted mime_type=image/png]" in main_branch["text"]
    assert "<thinking_level_change>" not in main_branch["text"]
    assert "<session_info>" not in main_branch["text"]
    assert "<compaction>" not in main_branch["text"]

    assert alternate_branch["id"] == "session-pi-001:label-leaf"
    assert alternate_branch["metadata"]["leaf_id"] == "label-leaf"
    assert alternate_branch["metadata"]["branch_depth"] == 10
    assert alternate_branch["metadata"]["filtered_event_count"] == 5
    assert alternate_branch["metadata"]["branch_summary_count"] == 1
    assert alternate_branch["metadata"]["custom_message_count"] == 1
    assert alternate_branch["metadata"]["tool_error_count"] == 1
    assert "<branch_summary>" in alternate_branch["text"]
    assert "<custom>" in alternate_branch["text"]
    assert "[custom_type=review-note]" in alternate_branch["text"]
    assert "[error]" in alternate_branch["text"]
    assert "<custom_message>" not in alternate_branch["text"]
    assert "<custom>" in alternate_branch["text"]
    assert "<label>" not in alternate_branch["text"]
    assert "<model_change>" not in alternate_branch["text"]


def test_convert_pi_session_to_dolma_end_to_end(tmp_path, read_all_jsonl_gz, validate_dolma_record) -> None:
    sample = _load_fixture("pi_mono_session_sample.json")
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "documents"
    input_dir.mkdir(parents=True, exist_ok=True)

    session_file = input_dir / sample["file_path"]
    session_file.write_text(
        "\n".join(json.dumps(trace) for trace in sample["traces"]) + "\n",
        encoding="utf-8",
    )

    convert_pi_session_to_dolma(
        PiSessionToDolmaConfig(
            input_path=str(input_dir),
            output_path=str(output_dir),
            input_glob="20*.jsonl",
            max_tool_output_chars=240,
        )
    )

    records = read_all_jsonl_gz(output_dir)

    assert len(records) == 2
    for record in records:
        validate_dolma_record(record)
        assert record["source"] == PI_MONO_SOURCE
        assert record["metadata"]["trace_source"] == PI_MONO_SOURCE
        assert record["metadata"]["session_id"] == sample["session_id"]

    texts = {record["id"]: record["text"] for record in records}
    assert (
        '<trace source="badlogicgames/pi-mono" trace_id="session-pi-001:compaction-leaf">'
        in texts["session-pi-001:compaction-leaf"]
    )
    assert (
        '<trace source="badlogicgames/pi-mono" trace_id="session-pi-001:label-leaf">'
        in texts["session-pi-001:label-leaf"]
    )
    assert "<branch_summary>" in texts["session-pi-001:label-leaf"]
    assert "<custom>" in texts["session-pi-001:label-leaf"]
    assert "<compaction>" not in texts["session-pi-001:compaction-leaf"]
