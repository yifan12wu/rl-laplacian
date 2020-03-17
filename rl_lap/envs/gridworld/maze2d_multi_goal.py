import numpy as np
import collections

from . import maze2d_base

ObservationType = collections.namedtuple('ObservationType', 'agent, goal')

class Maze2DMultiGoal(maze2d_base.Maze2DBase):

    def __init__(
            self, 
            maze, 
            episode_len=50, 
            start_pos='first', 
            use_stay_action=True,
            reward_type='pos',
            goal_pos=None,
            goal_rewards=None,
            end_at_goal=False):
        super(Maze2DMultiGoal, self).__init__(
                maze=maze,
                episode_len=episode_len,
                start_pos=start_pos,
                use_stay_action=use_stay_action)
        assert reward_type in ['pos', 'neg']
        self._reward_type = reward_type 
        assert len(goal_pos) == len(goal_rewards)
        self._goal_pos = goal_pos
        self._goal_rewards = goal_rewards
        self._end_at_goal = end_at_goal
        # to be maintained during each episode
        self._last_reward = 0.0
        self._goal_achieved = False

    def begin_episode(self):
        super(Maze2DMultiGoal, self).begin_episode()
        self._last_reward = 0.0
        self._goal_achieved = False

    def step(self, action):
        super(Maze2DMultiGoal, self).step(action)
        goal_achieved = False
        for g, r in zip(self._goal_pos, self._goal_rewards):
            if maze2d_base.is_same_pos(self._agent_pos, g):
                self._last_reward = r
                goal_achieved = True
                if self._end_at_goal:
                    self._should_end_episode = True
        if not goal_achieved:
            if self._reward_type == 'pos':
                self._last_reward = 0.0
            else:
                self._last_reward = -1.0

    def get_observation(self):
        agent_obs = self.pos_to_obs(self._agent_pos)
        goal_obs = None
        # goal_obs = self.pos_to_obs(self._goal_pos)
        return ObservationType(agent=agent_obs, goal=goal_obs)

    def get_reward(self):
        return self._last_reward
