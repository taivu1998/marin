# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Research-first registry for transcript-style agent-trace mid-training datasets."""

from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize import lm_mixture_data_config
from marin.transform.agent_traces.opentraces_runtime_to_dolma import (
    OPENTRACES_RUNTIME_SOURCE,
    OpenTracesRuntimeToDolmaConfig,
    convert_opentraces_runtime_to_dolma,
)
from marin.transform.agent_traces.pi_session_to_dolma import (
    PI_MONO_SOURCE,
    PiSessionToDolmaConfig,
    convert_pi_session_to_dolma,
)

from experiments.defaults import default_download, default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.midtraining_datasets import finemath_3_plus_tokenized

OPENTRACES_RUNTIME_REVISION = "778faed3c7d7a36089324f4f3f5bab66bf77f2ad"
PI_MONO_REVISION = "dac2a1d3ba12dda597b973a791a77618ccb5f413"

OPENTRACES_RUNTIME_TOKEN_RATIO = 0.01
PI_MONO_TOKEN_RATIO = 0.005
FINEMATH_BASELINE_TOKEN_RATIO = 1.0 - OPENTRACES_RUNTIME_TOKEN_RATIO - PI_MONO_TOKEN_RATIO

opentraces_runtime_raw = default_download(
    name="raw/opentraces_runtime",
    hf_dataset_id=OPENTRACES_RUNTIME_SOURCE,
    revision=OPENTRACES_RUNTIME_REVISION,
    override_output_path="raw/opentraces_runtime",
    hf_urls_glob=["data/traces_*.jsonl", "dataset_infos.json", "quality.json", "*.md"],
)

opentraces_runtime_documents = ExecutorStep(
    name="documents/opentraces_runtime",
    fn=convert_opentraces_runtime_to_dolma,
    config=OpenTracesRuntimeToDolmaConfig(
        input_path=opentraces_runtime_raw,
        output_path=this_output_path(),
    ),
)

opentraces_runtime_tokenized = default_tokenize(
    name="opentraces_runtime",
    dataset=opentraces_runtime_documents,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="text"),
)

pi_mono_raw = default_download(
    name="raw/pi_mono",
    hf_dataset_id=PI_MONO_SOURCE,
    revision=PI_MONO_REVISION,
    override_output_path="raw/pi_mono",
    hf_urls_glob=["*.jsonl", "*.md"],
)

pi_mono_documents = ExecutorStep(
    name="documents/pi_mono",
    fn=convert_pi_session_to_dolma,
    config=PiSessionToDolmaConfig(
        input_path=pi_mono_raw,
        output_path=this_output_path(),
    ),
)

pi_mono_tokenized = default_tokenize(
    name="pi_mono",
    dataset=pi_mono_documents,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="text"),
)

agent_trace_midtraining_components = {
    "finemath_3_plus": finemath_3_plus_tokenized,
    "opentraces_runtime": opentraces_runtime_tokenized,
    "pi_mono": pi_mono_tokenized,
}

agent_trace_midtraining_weights = {
    "finemath_3_plus": FINEMATH_BASELINE_TOKEN_RATIO,
    "opentraces_runtime": OPENTRACES_RUNTIME_TOKEN_RATIO,
    "pi_mono": PI_MONO_TOKEN_RATIO,
}

agent_trace_midtraining_mixture = lm_mixture_data_config(
    components=agent_trace_midtraining_components,
    weights=agent_trace_midtraining_weights,
)
