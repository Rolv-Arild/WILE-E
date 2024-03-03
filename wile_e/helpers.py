"""Helpers for RL environment."""

import functools
from typing import Any, Dict, NamedTuple, Sequence

from acme import specs
from acme.agents.jax import ppo
from acme.agents.jax.multiagent.decentralized import factories
from acme.jax import networks as networks_lib
from acme.jax import utils as acme_jax_utils
from acme.multiagent import types as ma_types
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class CategoricalParams(NamedTuple):
    """Parameters for a categorical distribution."""
    logits: jnp.ndarray


def make_rl_ppo_networks(
        environment_spec: specs.EnvironmentSpec,
        hidden_layer_sizes: Sequence[int] = (512, 512, 512, 512),
) -> ppo.PPONetworks:
    """Returns PPO networks used by the agent in the rl environments."""

    # Check that rl environment is defined with discrete actions, 0-indexed
    assert np.issubdtype(environment_spec.actions.dtype, np.integer), (
        'Expected rl environment to have discrete actions with int dtype'
        f' but environment_spec.actions.dtype == {environment_spec.actions.dtype}'
    )
    assert environment_spec.actions.minimum == 0, (
        'Expected rl environment to have 0-indexed action indices, but'
        f' environment_spec.actions.minimum == {environment_spec.actions.minimum}'
    )
    num_actions = environment_spec.actions.maximum + 1

    def forward_fn(inputs):
        processed_inputs = inputs
        trunk = hk.nets.MLP(hidden_layer_sizes, activation=jnp.tanh)
        h = trunk(processed_inputs)
        logits = hk.Linear(num_actions)(h)
        values = hk.Linear(1)(h)
        values = jnp.squeeze(values, axis=-1)
        return (CategoricalParams(logits=logits), values)

    # Transform into pure functions.
    forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

    dummy_obs = acme_jax_utils.zeros_like(environment_spec.observations)
    dummy_obs = acme_jax_utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
    network = networks_lib.FeedForwardNetwork(
        lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply)
    return make_categorical_ppo_networks(network)  # pylint:disable=undefined-variable


def make_categorical_ppo_networks(
        network: networks_lib.FeedForwardNetwork) -> ppo.PPONetworks:
    """Constructs a PPONetworks for Categorical Policy from FeedForwardNetwork.

    Args:
      network: a transformed Haiku network (or equivalent in other libraries) that
        takes in observations and returns the action distribution and value.

    Returns:
      A PPONetworks instance with pure functions wrapping the input network.
    """

    def log_prob(params: CategoricalParams, action):
        return tfd.Categorical(logits=params.logits).log_prob(action)

    def entropy(params: CategoricalParams, key: networks_lib.PRNGKey):
        del key
        return tfd.Categorical(logits=params.logits).entropy()

    def sample(params: CategoricalParams, key: networks_lib.PRNGKey):
        return tfd.Categorical(logits=params.logits).sample(seed=key)

    def sample_eval(params: CategoricalParams, key: networks_lib.PRNGKey):
        del key
        return tfd.Categorical(logits=params.logits).mode()

    return ppo.PPONetworks(
        network=network,
        log_prob=log_prob,
        entropy=entropy,
        sample=sample,
        sample_eval=sample_eval)


def init_default_rl_network(
        agent_type: str,
        agent_spec: specs.EnvironmentSpec) -> ma_types.Networks:
    """Returns default networks for rl environment."""
    if agent_type == factories.DefaultSupportedAgent.PPO:
        return make_rl_ppo_networks(agent_spec)
    else:
        raise ValueError(f'Unsupported agent type: {agent_type}.')
