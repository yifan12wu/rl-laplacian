import numpy as np
import collections

from . import maze2d_base

ObservationType = collections.namedtuple('ObservationType', 'agent, goal')

class Maze2DSingleGoal(maze2d_base.Maze2DBase):

    def __init__(
            self, 
            maze, 
            episode_len=50, 
            start_pos='first', 
            use_stay_action=True,
            reward_type='neg',
            goal_pos=None,
            end_at_goal=False):
        super(Maze2DSingleGoal, self).__init__(
                maze=maze,
                episode_len=episode_len,
                start_pos=start_pos,
                use_stay_action=use_stay_action)
        self._goal_pos = goal_pos
        self._rand_goal_pos = (self._goal_pos is None)
        assert reward_type in ['neg', 'pos']
        self._reward_type = reward_type
        if reward_type == 'neg':
            self._rewards = (-1.0, 0.0)
        elif reward_type == 'pos':
            self._rewards = (0.0, 1.0)
        self._end_at_goal = end_at_goal
        # to be maintained during each episode
        self._last_reward = 0.0
        self._goal_achieved = False

    def begin_episode(self):
        super(Maze2DSingleGoal, self).begin_episode()
        # set goal pos if random
        if self._rand_goal_pos:
            self._goal_pos = self._maze.random_empty_grids(1)[0]
        self._last_reward = 0.0
        self._goal_achieved = False

    def step(self, action):
        super(Maze2DSingleGoal, self).step(action)
        assert self._goal_pos is not None
        if maze2d_base.is_same_pos(self._agent_pos, self._goal_pos):
            self._last_reward = self._rewards[1]
            if self._end_at_goal:
                self._should_end_episode = True
        else:
            self._last_reward = self._rewards[0]


    def get_observation(self):
        agent_obs = self.pos_to_obs(self._agent_pos)
        goal_obs = self.pos_to_obs(self._goal_pos)
        return ObservationType(agent=agent_obs, goal=goal_obs)

    def get_reward(self):
        return self._last_reward

    def pos_to_full_obs(self, pos):
        obs = self.pos_to_obs(pos)
        return ObservationType(agent=obs, goal=obs)

