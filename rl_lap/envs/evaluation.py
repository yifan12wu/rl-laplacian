import numpy as np


class BasicEvaluator(object):

    def __init__(self, env_factory, max_ep_len=10000, rshift=0.0):
        self._env_factory = env_factory
        self._max_ep_len = max_ep_len
        self._rshift = rshift

    def run_test(self, n_episodes, policy_fn):
        n = n_episodes
        env = self._env_factory()
        r = np.zeros([n])
        for i in range(n):
            r_tmp = 0
            ts = env.reset()
            step = 0
            context = None
            while (not env.is_end_episode
                   and step <= self._max_ep_len):
                state = (ts, context)
                a, context = policy_fn(state)
                ts = env.step(a)
                r_tmp += (ts.reward + self._rshift)
            r[i] = r_tmp
        r_mean = np.mean(r)
        r_std = np.std(r)
        return r_mean, r_std


