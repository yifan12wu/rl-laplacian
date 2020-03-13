import numpy as np
import collections


# H: horizon, number of transitions.
# h: 0,...,H.
# r: episodic return.
EpisodicStep = collections.namedtuple('EpisodicStep', 'step, h, H, r')


class EpisodicReplayBuffer:
    '''Only store full episodes.'''

    def __init__(self, max_size):
        self._max_size = max_size
        self._current_size = 0
        self._current_idx = 0
        self._episodes = []
        self._episode_buffer = []
        self._r = 0.0
        # other indexing for sampling
        self._steps = []
        self._step1s = []
        self._step2s = []

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
                H = len(self._episode_buffer) - 1
                for h in range(H + 1):
                    epi_step = EpisodicStep(self._episode_buffer[i], 
                            h, H, self._r)
                    episode.append(epi_step)
                # save as data
                # TODO: decide when to overwrite old episodes
                self._episodes.append(episode)
                self._current_size += H
                # refresh episode buffer
                self._episode_buffer = []
                self._r = 0.0
            '''
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
            '''

    def sample_transitions(self, batch_size):
        indices = np.random.choice(self._size, batch_size, replace=True)
        s1 = list([self._step_1[idx] for idx in indices])
        s2 = list([self._step_2[idx] for idx in indices])
        return s1, s2

    def sample_steps(self, batch_size):
        indices = np.random.choice(self._size, batch_size, replace=True)
        s = list([self._step_2[idx] for idx in indices])
        return s



    @property
    def current_size(self):
        return self._size

    @property
    def max_size(self):
        return self._max_size

    def sample_pairs_in_episodes(
            self, batch_size, discount=0.0, max_range=200):
        sample_prob = np.array([(1.0 - discount) * np.power(discount, i)
                for i in range(max_range)])


  def sample_positive(self, batch_size, discount=0.9, max_range=1):
    """return two lists of timesteps, leave batch aggregation to the learner,
    so that the learner can decide which part of the timestep to use."""
    self.set_max_range(max_range)
    self.set_discount(discount)
    trj_indices = np.random.choice(self._size, batch_size, replace=True)
    intervals = np.random.choice(self._max_range, batch_size,
                                 replace=True, p=self._sample_prob) + 1
    s1 = []
    s2 = []
    for trj_idx, intv in zip(trj_indices, intervals):
      s1_idx = np.random.randint(len(self._data[trj_idx])-intv)
      s1.append(self._data[trj_idx][s1_idx])
      s2.append(self._data[trj_idx][s1_idx+intv])
    return s1, s2

  def sample_transitions(self, batch_size):
    trj_indices = np.random.choice(self._size, batch_size, replace=True)
    s1 = []
    s2 = []
    for trj_idx in trj_indices:
      s1_idx = np.random.randint(len(self._data[trj_idx])-1)
      s1.append(self._data[trj_idx][s1_idx])
      s2.append(self._data[trj_idx][s1_idx+1])
    return s1, s2

  def sample_steps(self, batch_size):
    trj_indices = np.random.choice(self._size, batch_size, replace=True)
    s = []
    for trj_idx in trj_indices:
      s_idx = np.random.randint(len(self._data[trj_idx]))
      s.append(self._data[trj_idx][s_idx])
    return s

  def sample_transitions_with_goal(self, batch_size):
    """
    goal:
        random: randomly sample a goal between s_{t+1} and s_T as sg
        last: set s_T as sg
    """
    trj_indices = np.random.choice(self._size, batch_size, replace=True)
    intervals = np.random.choice(self._max_range, batch_size,
                                 replace=True, p=self._sample_prob) + 1
    s1 = []
    s2 = []
    sg = []
    for trj_idx, intv in zip(trj_indices, intervals):
      s1_idx = np.random.randint(len(self._data[trj_idx])-intv)
      s1.append(self._data[trj_idx][s1_idx])
      sg.append(self._data[trj_idx][s1_idx+intv])
      s2.append(self._data[trj_idx][s1_idx+1])
    return s1, s2, sg

  def sample_positive_with_next(self, batch_size):
    """return 3 lists of timesteps, leave batch aggregation to the learner,
    so that the learner can decide which part of the timestep to use."""
    trj_indices = np.random.choice(self._size, batch_size, replace=True)
    intervals = np.random.choice(self._max_range, batch_size,
                                 replace=True, p=self._sample_prob) + 1
    s1 = []
    s2 = []
    s_next = []
    for trj_idx, intv in zip(trj_indices, intervals):
      s1_idx = np.random.randint(len(self._data[trj_idx])-intv)
      s1.append(self._data[trj_idx][s1_idx])
      s2.append(self._data[trj_idx][s1_idx+intv])
      s_next.append(self._data[trj_idx][s1_idx+1])
    return s1, s2, s_next

  def sample_negative(self, batch_size):
    trj_indices = np.random.choice(self._size, size=[batch_size, 2], replace=True)
    s1 = []
    s2 = []
    for trj_idx_pair in trj_indices:
      s1_idx = np.random.randint(len(self._data[trj_idx_pair[0]]))
      s2_idx = np.random.randint(len(self._data[trj_idx_pair[1]]))
      s1.append(self._data[trj_idx_pair[0]][s1_idx])
      s2.append(self._data[trj_idx_pair[1]][s2_idx])
    return s1, s2