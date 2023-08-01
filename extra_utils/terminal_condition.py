"""
A module containing implementations of common terminal conditions.
"""

from rlgym_sim.utils.terminal_conditions import TerminalCondition
from rlgym_sim.utils.gamestates import GameState


class GoalScoredCondition(TerminalCondition):
    """
    A condition that will terminate an episode as soon as a goal is scored by either side.
    """

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        """
        Check to see if the game score for either team has been changed. If either score has changed, update the current
        known scores for both teams and return `True`. Note that the known game scores are never reset for this object
        because the game score is not set to 0 for both teams at the beginning of an episode.
        """
        ball_y = current_state.ball.position[1]
        return ball_y < (-5100) or ball_y > 5100
