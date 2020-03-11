import numpy as np
from ..envs import env_base

FIRST = env_base.StepType.FIRST
MID = env_base.StepType.MID
FINAL = env_base.StepType.FINAL

class ReplayBuffer(object):
    """Store trajectories, support positive and negative sampling."""

    def __init__(self, max_size):
        self._max_size = max_size
        self._curr_idx = 0
        self._size = 0
        self._step_1 = []
        self._step_2 = []
        self._prev_step = None

    def add_steps(self, steps):
        """
        steps: a list of (time_step, action, context tuples)
        """
        for step in steps:
            if self._prev_step is not None and step[0].step_type != FIRST:
                if self._size < self._max_size:
                    self._step_1.append(self._prev_step)
                    self._step_2.append(step)
                    self._size += 1
                else:
                    self._step_1[self._curr_idx] = self._prev_step
                    self._step_2[self._curr_idx] = step
            self._curr_idx = (self._curr_idx + 1) % self._max_size
            self._prev_step = step

    def sample_transitions(self, batch_size):
        indices = np.random.choice(self._size, batch_size, replace=True)
        s1 = list([self._step_1[idx] for idx in indices])
        s2 = list([self._step_2[idx] for idx in indices])
        return s1, s2

    def sample_steps(self, batch_size):
        indices = np.random.choice(self._size, batch_size, replace=True)
        s = list([self._step_2[idx] for idx in indices])
        return s


    def sample_n_steps(self, batch_size, n):
        indices = np.random.choice(self._size-n+1, batch_size, replace=True)
        s1 = list([self._step_1[idx:idx+n] for idx in indices])
        s2 = list([self._step_2[idx:idx+n] for idx in indices])
        return s1, s2

    @property
    def size(self):
        return self._size

    @property
    def max_size(self):
        return self._max_size
