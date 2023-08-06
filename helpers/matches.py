import numpy as np
from extra_utils.goal_scored_reward import GoalScoredReward
from extra_utils.terminal_condition import GoalScoredCondition
from rlgym_sim.envs import Match
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym_sim.utils.state_setters import RandomState
from helpers.extra_state_setters import (
    GoaliePracticeState,
    WeightedSampleSetter,
    WallPracticeState,
)
from helpers.rolv_replay_setter import ScoredReplaySetter
from rlgym_sim.utils.obs_builders import AdvancedObs
from helpers.lookup_action import LookupAction


def get_replay_setter_match(
    file_path: str, start_idx: int, end_idx, timeout: int = 100
):
    return Match(
        team_size=2,
        reward_function=GoalScoredReward(),
        spawn_opponents=True,
        obs_builder=AdvancedObs(),
        terminal_conditions=[
            GoalScoredCondition(),
            TimeoutCondition(timeout),
        ],
        state_setter=ScoredReplaySetter(file_path, start_idx, end_idx),
        action_parser=LookupAction(),
    )


def get_random_goalie_match(timeout: int = 300):
    return Match(
        team_size=2,
        reward_function=GoalScoredReward(),
        spawn_opponents=True,
        obs_builder=AdvancedObs(),
        terminal_conditions=[
            GoalScoredCondition(),
            TimeoutCondition(timeout),
        ],
        state_setter=WeightedSampleSetter(
            (
                RandomState(True, True, False),
                GoaliePracticeState(
                    allow_enemy_interference=True, first_defender_in_goal=True
                ),
                WallPracticeState(),
            ),
            (0.3, 0.5, 0.2),
        ),
        action_parser=LookupAction(),
    )
