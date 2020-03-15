import numpy as np
import collections


# H: horizon, number of transitions.
# h: 1,...,H.
# r: episodic return.
EpisodicStep = collections.namedtuple('EpisodicStep', 'step, h, H, r')


class EpisodicReplayBuffer:
    '''Only store full episodes.'''

    def __init__(self, max_size):
        self._max_size = max_size
        self._current_size = 0
        self._next_idx = 0
        self._episode_buffer = []
        self._r = 0.0
        self._episodes = []

    @property
    def current_size(self):
        return self._current_size

    @property
    def max_size(self):
        return self._max_size

    def add_steps(self, steps):
        '''
        steps: a list of Step(time_step, action, context).
        '''
        for step in steps:
            self._episode_buffer.append(step)
            self._r += step.time_step.reward
            if step.time_step.is_last:
                # construct a formal episode
                episode = []
                H = len(self._episode_buffer)
                for h in range(H):
                    epi_step = EpisodicStep(self._episode_buffer[i], 
                            h + 1, H, self._r)
                    episode.append(epi_step)
                # save as data
                if self._next_idx == self._current_size:
                    if self._current_size < self._max_size:
                        self._episodes.append(episode)
                        self._current_size += 1
                        self._next_idx += 1
                    else:
                        self._episodes[0] = episode
                        self._next_idx = 1
                else:
                    self._episodes[self._next_idx] = episode
                    self._next_idx += 1
                # refresh episode buffer
                self._episode_buffer = []
                self._r = 0.0

    def sample_transitions(self, batch_size):
        epi_indices = np.random.choice(
            self._current_size, batch_size, replace=True)
        s1 = []
        s2 = []
        for epi_idx in epi_indices:
            episode = self._episodes[epi_idx]
            i = np.random.randint(episode[0].step.H - 1)
            s1.append(episode[i])
            s2.append(episode[i + 1])
        return s1, s2

    def sample_steps(self, batch_size):
        epi_indices = np.random.choice(
            self._current_size, batch_size, replace=True)
        s = []
        for epi_idx in epi_indices:
            episode = self._episodes[epi_idx]
            i = np.random.randint(episode[0].step.H)
            s.append(episode[i])
        return s

    def sample_pairs(self, batch_size, discount=0.0):
        epi_indices = np.random.choice(
            self._current_size, batch_size, replace=True)
        s1 = []
        s2 = []
        for epi_idx in epi_indices:
            episode = self._episodes[epi_idx]
            H = episodes[0]step.H
            sample_distr = np.ones(H - 1)
            for i in range(H - 1):
                sample_distr[i] = np.power(discount, i)
            interval = np.random.choice(H - 1, p=sample_distr)
            i = np.random.randint(H - interval - 1)
            s1.append(episode[i])
            s2.append(episode[i + interval + 1])
        return s1, s2
