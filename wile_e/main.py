# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example running PPO on continuous control tasks."""

from absl import flags
from acme.agents.jax import ppo
from dm_env import specs
from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import TouchReward, CombinedReward, GoalReward
from rlgym.rocket_league.state_mutators import KickoffMutator, FixedTeamSizeMutator

import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

from wile_e.config_objects import VelocityPlayerToBallReward
from wile_e.dm_rl import RocketLeague

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
                             'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'control:cartpole:balance', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 100_000_000_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 1_000_000, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
flags.DEFINE_integer('num_distributed_actors', 64,
                     'Number of actors to use in the distributed setting.')


def make_environment():
    state_mutators = [FixedTeamSizeMutator(), KickoffMutator()]
    obs_builder = DefaultObs(zero_padding=1)
    tick_skip = 8
    action_parser = RepeatAction(LookupTableAction(), tick_skip)
    reward_function = CombinedReward(
        (VelocityPlayerToBallReward(), tick_skip / 120),
        (TouchReward(), tick_skip / 120),
        (GoalReward(), 60)
    )
    terminal_conditions = [GoalCondition()]
    truncation_conditions = [NoTouchTimeoutCondition(60), TimeoutCondition(5 * 60)]
    env = RocketLeague(state_mutators, obs_builder, action_parser, reward_function,
                       terminal_conditions, truncation_conditions, discount=0.995,
                       action_spec=specs.DiscreteArray(90),
                       observation_spec=specs.Array((obs_builder.get_obs_space(None),), dtype=float))
    return env


def build_experiment_config():
    """Builds PPO experiment config which can be executed in different ways."""
    # Create an environment, grab the spec, and use it to create networks.
    config = ppo.PPOConfig(
        normalize_advantage=True,
        normalize_value=True,
        obs_normalization_fns_factory=ppo.build_mean_std_normalizer)
    ppo_builder = ppo.PPOBuilder(config)

    layer_sizes = (256, 256, 256)
    return experiments.ExperimentConfig(
        builder=ppo_builder,
        environment_factory=make_environment,  # lambda seed: helpers.make_environment(suite, task),
        network_factory=lambda spec: ppo.make_networks(spec, layer_sizes),
        seed=FLAGS.seed,
        max_num_actor_steps=FLAGS.num_steps)


def main(_):
    config = build_experiment_config()
    if FLAGS.run_distributed:
        program = experiments.make_distributed_experiment(
            experiment=config, num_actors=FLAGS.num_distributed_actors)
        lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
    else:
        experiments.run_experiment(
            experiment=config,
            eval_every=FLAGS.eval_every,
            num_eval_episodes=FLAGS.evaluation_episodes)


if __name__ == '__main__':
    app.run(main)
