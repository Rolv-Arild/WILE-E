from typing import List

import dm_env
from rlgym.api import StateMutator, ObsBuilder, ActionParser, RewardFunction, DoneCondition
from rlgym.rocket_league.sim.rocketsim_engine import RocketSimEngine
from dm_env import TimeStep, StepType


class RocketLeague(dm_env.Environment):
    def __init__(
            self,
            state_mutators: List[StateMutator],
            obs_builder: ObsBuilder,
            action_parser: ActionParser,
            reward_function: RewardFunction,
            terminal_conditions: List[DoneCondition],
            truncation_conditions: List[DoneCondition],
            discount: float = 1.0,
            action_spec=None,
            observation_spec=None,
    ):
        self.engine = RocketSimEngine()
        self.state_mutators = state_mutators
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.reward_function = reward_function
        self.terminal_conditions = terminal_conditions
        self.truncation_conditions = truncation_conditions
        self.discount = discount
        self.shared_info = {}

        self.num_agents = 2
        self.agents = ["blue_0", "orange_0"]

        self._action_spec = action_spec
        self._observation_spec = observation_spec

    def reset(self) -> TimeStep:
        self.shared_info = {}
        base_state = self.engine.create_base_state()
        for mutator in self.state_mutators:
            mutator.apply(base_state, self.shared_info)
        agents = list(base_state.cars.keys())
        self.num_agents = len(agents)
        self.agents = agents

        self.engine.set_state(base_state, self.shared_info)

        self.obs_builder.reset(base_state, self.shared_info)
        self.action_parser.reset(base_state, self.shared_info)
        self.reward_function.reset(base_state, self.shared_info)
        for condition in self.terminal_conditions:
            condition.reset(base_state, self.shared_info)
        for condition in self.truncation_conditions:
            condition.reset(base_state, self.shared_info)

        observations = self.obs_builder.build_obs(agents, base_state, self.shared_info)
        return dm_env.restart(observations)

    def step(self, action) -> TimeStep:
        engine_actions = self.action_parser.parse_actions(action, self.engine.state, self.shared_info)
        state = self.engine.step(engine_actions, self.shared_info)
        agents = list(state.cars.keys())
        observations = self.obs_builder.build_obs(agents, state, self.shared_info)
        is_terminated = False
        for condition in self.terminal_conditions:
            agents_dones = condition.is_done(agents, state, self.shared_info)
            if any(agents_dones.values()):
                is_terminated = True
                break
        is_truncated = False
        if not is_terminated:
            for condition in self.truncation_conditions:
                agents_truncated = condition.is_done(agents, state, self.shared_info)
                if any(agents_truncated.values()):
                    is_truncated = True
                    break
        rewards = self.reward_function.get_rewards(agents, state,
                                                   {a: is_terminated for a in agents},
                                                   {a: is_truncated for a in agents},
                                                   self.shared_info)
        if is_terminated:
            return dm_env.termination(rewards, observations)
        if is_truncated:
            return dm_env.truncation(rewards, observations, self.discount)
        return dm_env.transition(rewards, observations, self.discount)

    def observation_spec(self):
        return {k: self._observation_spec for k in self.agents}

    def action_spec(self):
        return {k: self._action_spec for k in self.agents}

    def reward_spec(self):
        spec = super().reward_spec()
        return {k: spec for k in self.agents}

    def discount_spec(self):
        spec = super().discount_spec()
        return {k: spec for k in self.agents}
