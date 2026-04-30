# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pilot mid-training run that adds a small agent-trace transcript slice to a reasoning baseline."""

import dataclasses

from marin.execution.executor import executor_main

from experiments.defaults import default_train
from experiments.llama import llama_150m, llama_150m_train_config
from experiments.midtraining_agent_trace_datasets import agent_trace_midtraining_mixture

pilot_train_config = dataclasses.replace(
    llama_150m_train_config,
    num_train_steps=2_000,
    steps_per_eval=250,
    steps_per_export=500,
    warmup=0.05,
    decay=0.2,
)

agent_trace_midtraining_pilot = default_train(
    name="agent-trace-midtraining-pilot",
    tokenized=agent_trace_midtraining_mixture,
    model_config=llama_150m,
    train_config=pilot_train_config,
    tags=["agent-traces", "midtraining", "opentraces-runtime", "pi-mono", "pilot"],
    eval_harness_tasks=[],
    use_default_validation=False,
)


if __name__ == "__main__":
    executor_main(
        steps=[agent_trace_midtraining_pilot],
        description=(
            "Pilot mid-training run with a 98.5/1.0/0.5 FineMath/OpenTraces-runtime/Pi-Mono token mixture "
            "to validate the transcript-based agent-trace lane."
        ),
    )
