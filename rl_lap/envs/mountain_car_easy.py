# An easier mountaincar env with reward shaping
from gym.envs.classic_control import mountain_car


ENV_ID = 'MountainCarEasy-v0'


class MountainCarEasyEnv(mountain_car.MountainCarEnv):

    def step(self, action):
        obs, reward, done, info = super(MountainCarEasyEnv, self).step(action)
        if obs[0] >= -0.5:
            reward += ((obs[0] + 0.5) * 0.5)
        # if obs[0] >= 0.5:
        #   reward += 1
        return obs, reward, done, info

