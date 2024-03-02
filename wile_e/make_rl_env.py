from dm_env import specs
from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import TouchReward, CombinedReward, GoalReward
from rlgym.rocket_league.state_mutators import KickoffMutator, FixedTeamSizeMutator

from wile_e.config_objects import VelocityPlayerToBallReward, GoalViewReward
from wile_e.dm_rl import RocketLeague


def make_rl_environment(seed):
    state_mutators = [FixedTeamSizeMutator(), KickoffMutator()]
    obs_builder = DefaultObs(zero_padding=1)
    tick_skip = 8
    action_parser = RepeatAction(LookupTableAction(), tick_skip)
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
                       action_spec=specs.DiscreteArray(90),
                       observation_spec=specs.Array((obs_builder.get_obs_space(None),), dtype=float))
    return env
