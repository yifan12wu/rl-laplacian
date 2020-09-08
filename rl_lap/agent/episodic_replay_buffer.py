import numpy as np
import collections


# H: horizon, number of transitions.
# h: 1,...,H.
# r: episodic return.
EpisodicStep = collections.namedtuple('EpisodicStep', 'step, h, H, r')


def discounted_sampling(ranges, discount):
    """Draw samples from the discounted distribution over 0, ...., n - 1, 
    where n is a range. The input ranges is a batch of such n`s.

    The discounted distribution is defined as
    p(y = i) = (1 - discount) * discount^i / (1 - discount^n).

    This function implement inverse sampling. We first draw
    seeds from uniform[0, 1) then pass them through the inverse cdf
    floor[ log(1 - (1 - discount^n) * seeds) / log(discount) ]
    to get the samples.
    """
    assert np.min(ranges) >= 1
    assert discount >= 0 and discount <= 1
    seeds = np.random.uniform(size=ranges.shape)
    if discount == 0:
        samples = np.zeros_like(seeds, dtype=np.int64)
    elif discount == 1:
        samples = np.floor(seeds * ranges, dtype=np.int64)
    else:
        samples = (np.log(1 - (1 - np.power(discount, ranges)) * seeds) 
                / np.log(discount))
        samples = np.floor(samples, dtype=np.int64)
    return samples


class EpisodicReplayBuffer:
    """Only store full episodes.
    
    Sampling returns EpisodicStep objects.
    """

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
        """
        steps: a list of Step(time_step, action, context).
        """
        for step in steps:
            self._episode_buffer.append(step)
            self._r += step.time_step.reward
            # Push each step into the episode buffer until an end-of-episode
            # step is found. 
            # self._r is used to track the cumulative return in each episode.
            if step.time_step.is_last:
                # construct a formal episode
                episode = []
                H = len(self._episode_buffer)
                for h in range(H):
                    epi_step = EpisodicStep(self._episode_buffer[h], 
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
        epi_indices = np.random.randint(self._current_size, size=batch_size)
        s1 = []
        s2 = []
        for epi_idx in epi_indices:
            episode = self._episodes[epi_idx]
            i = np.random.randint(episode[0].step.H - 1)
            s1.append(episode[i])
            s2.append(episode[i + 1])
        return s1, s2

    def sample_steps(self, batch_size):
        epi_indices = np.random.randint(self._current_size, size=batch_size)
        s = []
        for epi_idx in epi_indices:
            episode = self._episodes[epi_idx]
            i = np.random.randint(episode[0].H)
            s.append(episode[i])
        return s

    def sample_pairs(self, batch_size, discount=0.0):
        epi_indices = np.random.choice(
            self._current_size, batch_size, replace=True)
        s1 = []
        s2 = []
        for epi_idx in epi_indices:
            episode = self._episodes[epi_idx]
            H = episode[0].H
            # '''
            # TODO: fast sampling
            sample_distr = np.ones(H - 1)
            for i in range(H - 1):
                sample_distr[i] = np.power(discount, i)
            sample_distr /= np.sum(sample_distr)
            interval = np.random.choice(H - 1, p=sample_distr)
            # '''
            # interval = np.random.choice(H - 1)
            i = np.random.randint(H - interval - 1)
            s1.append(episode[i])
            s2.append(episode[i + interval + 1])
        return s1, s2
