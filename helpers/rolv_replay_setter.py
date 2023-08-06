import numpy as np
from helpers.replay_setter import ReplaySetter


class ScoredReplaySetter(ReplaySetter):
    def __init__(self, npz_arrays_file, start_idx: int = 0, end_idx: int = 0):
        if isinstance(npz_arrays_file, str):
            npz_arrays_file = np.load(npz_arrays_file)
        states = npz_arrays_file["states"]
        scores = npz_arrays_file["scores"]
        if start_idx < end_idx:
            if end_idx > len(scores):
                end_idx = len(scores)
            states = states[start_idx:end_idx]
            scores = scores[start_idx:end_idx]
        mask = ~np.isnan(states).any(axis=1) & ~np.isnan(scores)
        states = states[mask]
        scores = scores[mask]
        self.scores = scores
        super().__init__(states)

    def generate_probabilities(self):
        scores = self.scores**2  # Just to emphasize high scores more
        probs = scores / scores.sum()
        return probs
