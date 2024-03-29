import os
from typing import Callable, Dict

from absl import flags

from acme import specs
from acme.agents.jax.multiagent import decentralized
from absl import app
from acme.agents.jax.ppo import PPOConfig

import helpers
from acme.jax import experiments
from acme.jax import types as jax_types
from acme.multiagent import types as ma_types
from acme.utils import lp_utils
import dm_env
import launchpad as lp

from wile_e.make_rl_env import make_rl_environment

FLAGS = flags.FLAGS
_RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
                             'way. If False, will run single-threaded.')
_NUM_STEPS = flags.DEFINE_integer('num_steps', 1_000_000_000,
                                  'Number of env steps to run training for.')
_EVAL_EVERY = flags.DEFINE_integer('eval_every', 1_000_000,
                                   'How often to run evaluation.')
_ENV_NAME = flags.DEFINE_string('env_name', 'Rocket-League',
                                'What environment to run.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 100_000, 'Batch size.')
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')


def _make_environment_factory(env_name: str) -> jax_types.EnvironmentFactory:
    def environment_factory(seed: int) -> dm_env.Environment:
        return make_rl_environment(seed)

    return environment_factory


def _make_network_factory(
        agent_types: Dict[ma_types.AgentID, ma_types.GenericAgent]
) -> Callable[[specs.EnvironmentSpec], ma_types.MultiAgentNetworks]:
    def environment_factory(
            environment_spec: specs.EnvironmentSpec) -> ma_types.MultiAgentNetworks:
        return decentralized.network_factory(environment_spec, agent_types,
                                             helpers.init_default_rl_network)

    return environment_factory


def build_experiment_config() -> experiments.ExperimentConfig[
    ma_types.MultiAgentNetworks, ma_types.MultiAgentPolicyNetworks,
    ma_types.MultiAgentSample]:
    """Returns a config for rl experiments."""

    environment_factory = _make_environment_factory(_ENV_NAME.value)
    environment = environment_factory(_SEED.value)
    agent_types = {
        a: decentralized.DefaultSupportedAgent.PPO
        for a in environment.agents  # pytype: disable=attribute-error
    }
    # Example of how to set custom sub-agent configurations.
    ppo_configs = {'num_epochs': 2, 'unroll_length': 30}
    config_overrides = {
        k: ppo_configs for k, v in agent_types.items() if v == 'ppo'
    }

    configs = decentralized.default_config_factory(agent_types, _BATCH_SIZE.value,
                                                   config_overrides)

    builder = decentralized.DecentralizedMultiAgentBuilder(
        agent_types=agent_types, agent_configs=configs)

    return experiments.ExperimentConfig(
        builder=builder,
        environment_factory=environment_factory,
        network_factory=_make_network_factory(agent_types=agent_types),
        seed=_SEED.value,
        max_num_actor_steps=_NUM_STEPS.value)


def main(_):
    config = build_experiment_config()
    if _RUN_DISTRIBUTED.value:
        program = experiments.make_distributed_experiment(
            experiment=config, num_actors=os.cpu_count())
        lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
    else:
        experiments.run_experiment(
            experiment=config, eval_every=_EVAL_EVERY.value, num_eval_episodes=5)


if __name__ == '__main__':
    app.run(main)
