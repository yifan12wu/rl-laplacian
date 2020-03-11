import gym
from gym.envs import registration

from . import mountain_car_easy

registration.register(
    id='MountainCarEasy-v0',
    entry_point=mountain_car_easy.MountainCarEasyEnv,
    max_episode_steps=200)
