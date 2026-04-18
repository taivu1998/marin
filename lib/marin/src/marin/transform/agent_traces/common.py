# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared types for agent-trace transcript rendering."""

import dataclasses
from enum import StrEnum
from typing import TypeAlias

TraceTranscriptMetadataValue: TypeAlias = str | int | float | bool | None


class TraceSourceFormat(StrEnum):
    """Supported source schemas for transcript rendering."""

    OPENTRACES_RUNTIME = "opentraces_runtime"
    PI_SESSION = "pi_session"


class TraceBranchPolicy(StrEnum):
    """How one source trace should be segmented into transcript documents."""

    WHOLE_TRACE = "whole_trace"
    ROOT_TO_LEAF = "root_to_leaf"


@dataclasses.dataclass(frozen=True)
class TraceTranscriptConfig:
    """Configuration shared by transcript-producing transforms."""

    input_path: str
    output_path: str
    source_format: TraceSourceFormat
    max_tool_output_chars: int = 4000
    branch_policy: TraceBranchPolicy = TraceBranchPolicy.WHOLE_TRACE

    def __post_init__(self) -> None:
        if not self.input_path:
            raise ValueError("input_path must be non-empty.")
        if not self.output_path:
            raise ValueError("output_path must be non-empty.")
        if self.max_tool_output_chars <= 0:
            raise ValueError("max_tool_output_chars must be positive.")


@dataclasses.dataclass(frozen=True)
class TraceTranscriptRow:
    """Canonical row emitted by transcript-based mid-training transforms."""

    id: str
    source: str
    text: str
    metadata: dict[str, TraceTranscriptMetadataValue]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id must be non-empty.")
        if not self.source:
            raise ValueError("source must be non-empty.")
        if not self.text:
            raise ValueError("text must be non-empty.")

        invalid_keys = [
            key for key, value in self.metadata.items() if not isinstance(value, (str, int, float, bool, type(None)))
        ]
        if invalid_keys:
            raise ValueError(f"metadata values must be scalar. Invalid keys: {invalid_keys}")

    def to_dict(self) -> dict[str, str | dict[str, TraceTranscriptMetadataValue]]:
        """Convert the row into the Dolma-style dictionary expected by downstream steps."""
        return {
            "id": self.id,
            "source": self.source,
            "text": self.text,
            "metadata": dict(self.metadata),
        }
