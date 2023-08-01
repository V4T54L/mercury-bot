from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils import math
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np
from rlgym_sim.utils.common_values import ORANGE_TEAM


class GoalScoredReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return -1e-4

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == ORANGE_TEAM:
            ball = state.inverted_ball
        else:
            ball = state.ball

        return int(ball.position[1] > 5050) - int(ball.position[1] < -5050) * 0.8
