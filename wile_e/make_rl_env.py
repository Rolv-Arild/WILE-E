from typing import Dict, Any

import numpy as np
from dm_env import specs
from rlgym.api import StateMutator, StateType, ActionParser, AgentID, SpaceType
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import TouchReward, CombinedReward, GoalReward
from rlgym.rocket_league.state_mutators import KickoffMutator, FixedTeamSizeMutator

from wile_e.config_objects import VelocityPlayerToBallReward, GoalViewReward
from wile_e.dm_rl import RocketLeague


class FixKeyMutator(StateMutator[GameState]):
    # Acme uses '-' as a separator for learner names, so we need to replace it with '_'
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        for key in list(state.cars.keys()):
            state.cars[key.replace("-", "_")] = state.cars.pop(key)


class FixedLookupTableAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, int]):
    """
    World-famous discrete action parser which uses a lookup table to reduce the number of possible actions from 1944 to 90
    """

    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    def get_action_space(self, agent: AgentID) -> int:
        return len(self._lookup_table)

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def parse_actions(self, actions: Dict[AgentID, np.ndarray], state: GameState, shared_info: Dict[str, Any]) -> Dict[
        AgentID, np.ndarray]:
        parsed_actions = {}
        for agent, action in actions.items():
            parsed_actions[agent] = self._lookup_table[action]

        return parsed_actions

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])

        return np.array(actions)


class FixedRepeatAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, SpaceType]):
    """
    A simple wrapper to emulate tick skip
    """

    def __init__(self,
                 parser: ActionParser[AgentID, np.ndarray, np.ndarray, GameState, SpaceType],
                 repeats=8):
        super().__init__()
        self.parser = parser
        self.repeats = repeats

    def get_action_space(self, agent: AgentID) -> SpaceType:
        return self.parser.get_action_space(agent)

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def parse_actions(self, actions: Dict[AgentID, np.ndarray], state: GameState, shared_info: Dict[str, Any]) -> Dict[
        AgentID, np.ndarray]:
        parsed_actions = self.parser.parse_actions(actions, state, shared_info)
        repeat_actions = {}
        for agent, action in parsed_actions.items():
            if action.shape == (8,):
                action = np.expand_dims(action, axis=0)
            elif action.shape != (1, 8):
                raise ValueError(f"Expected action to have shape (8,) or (1,8), got {action.shape}")

            repeat_actions[agent] = action.repeat(self.repeats, axis=0)

        return repeat_actions


def make_rl_environment(seed):
    state_mutators = [FixedTeamSizeMutator(), KickoffMutator(), FixKeyMutator()]
    obs_builder = DefaultObs(zero_padding=1)
    tick_skip = 8
    action_parser = FixedRepeatAction(FixedLookupTableAction(), tick_skip)
    reward_function = CombinedReward(
        (VelocityPlayerToBallReward(), tick_skip / 120),
        (TouchReward(), tick_skip / 120),
        (GoalViewReward(), 1),
        (GoalReward(), 10)
    )
    terminal_conditions = [GoalCondition()]
    truncation_conditions = [NoTouchTimeoutCondition(60), TimeoutCondition(5 * 60)]
    env = RocketLeague(state_mutators, obs_builder, action_parser, reward_function,
                       terminal_conditions, truncation_conditions, discount=0.995,
                       action_spec=specs.DiscreteArray(90, name="action_index"),
                       observation_spec=specs.Array((obs_builder.get_obs_space(None),),
                                                    dtype=float,
                                                    name="observation"))
    return env
