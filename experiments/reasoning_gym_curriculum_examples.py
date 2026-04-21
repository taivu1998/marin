# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small curriculum examples for Prime-backed Reasoning Gym RL lessons."""

from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.rl_experiment_utils import RLExperimentConfig

DEFAULT_REASONING_GYM_EVAL_N_EXAMPLES = 128
REASONING_GYM_LEG_COUNTING_SEED = 42


def build_leg_counting_curriculum(
    run_id: str,
    config: RLExperimentConfig,
    eval_frequency: int,
) -> CurriculumConfig:
    """Build a minimal single-lesson Prime Reasoning Gym curriculum example."""
    sampling_params = SamplingParams(
        temperature=1.0,
        n_prompts=config.n_prompts,
        n_generations_per_prompt=config.n_generations_per_prompt,
        max_output_tokens=config.max_output_tokens,
        top_k=config.inference_top_k,
        stop_tokens=None,
    )

    return CurriculumConfig(
        lessons={
            "rg_leg_counting": LessonConfig(
                lesson_id="rg_leg_counting",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.prime_reasoning_gym_env.PrimeReasoningGymEnv",
                    env_args={
                        "gym": "leg_counting",
                        "num_train_examples": 10_000,
                        "num_eval_examples": DEFAULT_REASONING_GYM_EVAL_N_EXAMPLES,
                        "seed": REASONING_GYM_LEG_COUNTING_SEED,
                        "success_threshold": 1.0,
                    },
                ),
                dependencies=[],
                sampling_params=sampling_params,
            ),
        },
        eval_frequency=eval_frequency,
        micro_eval_frequency=None,
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=DEFAULT_REASONING_GYM_EVAL_N_EXAMPLES,
        max_seq_len=config.max_input_tokens + config.max_output_tokens,
    )
