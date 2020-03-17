import numpy as np
from . import maze
from . import maze2d_single_goal
from . import maze2d_multi_goal
from .. import env_base


FOUR_ROOM_MAZE = maze.Maze(maze.FourRoomsFactory(size=9))
FOUR_ROOM_GOAL_POS = np.array([15, 15])
FOUR_ROOM_START_POS = np.array([5, 5])
FOUR_ROOM_MULTIGOAL_POS = [np.array([1, 19]), np.array([19, 1]), np.array([19, 19])]
FOUR_ROOM_MULTIGOAL_REWARDS = [1.0, 1.0, 10.0]

ONE_ROOM_MAZE = maze.Maze(maze.SquareRoomFactory(size=15))
TWO_ROOM_MAZE = maze.Maze(maze.TwoRoomsFactory(size=15))

class FourRoomSingleGoal(env_base.Environment):
    def __init__(self, reward_type='neg', end_at_goal=False,
            goal_pos=FOUR_ROOM_GOAL_POS):
        task = maze2d_single_goal.Maze2DSingleGoal(
            maze=FOUR_ROOM_MAZE,
            episode_len=50,
            start_pos='random',
            use_stay_action=True,
            reward_type=reward_type,
            goal_pos=goal_pos,
            end_at_goal=end_at_goal)
        super(FourRoomSingleGoal, self).__init__(task)


class FourRoomSingleGoalSparse(env_base.Environment):
    def __init__(self, reward_type='neg', end_at_goal=False):
        task = maze2d_single_goal.Maze2DSingleGoal(
            maze=FOUR_ROOM_MAZE,
            episode_len=50,
            start_pos=FOUR_ROOM_START_POS,
            use_stay_action=True,
            reward_type=reward_type,
            goal_pos=FOUR_ROOM_GOAL_POS,
            end_at_goal=end_at_goal)
        super(FourRoomSingleGoalSparse, self).__init__(task)




class FourRoomRandomGoalFixedStart(env_base.Environment):
    def __init__(self, reward_type='neg', end_at_goal=False):
        task = maze2d_single_goal.Maze2DSingleGoal(
            maze=FOUR_ROOM_MAZE,
            episode_len=50,
            start_pos=FOUR_ROOM_START_POS,
            use_stay_action=True,
            reward_type=reward_type,
            goal_pos=None,
            end_at_goal=end_at_goal)
        super(FourRoomRandomGoalFixedStart, self).__init__(task)


class FourRoomMultiGoal(env_base.Environment):
    def __init__(self, reward_type='neg', end_at_goal=False):
        task = maze2d_multi_goal.Maze2DMultiGoal(
            maze=FOUR_ROOM_MAZE,
            episode_len=60,
            start_pos=FOUR_ROOM_START_POS,
            use_stay_action=True,
            reward_type=reward_type,
            goal_pos=FOUR_ROOM_MULTIGOAL_POS,
            goal_rewards=FOUR_ROOM_MULTIGOAL_REWARDS,
            end_at_goal=end_at_goal)
        super(FourRoomMultiGoal, self).__init__(task)


class FourRoomMultiGoalRandomStart(env_base.Environment):
    def __init__(self, reward_type='neg', end_at_goal=False):
        task = maze2d_multi_goal.Maze2DMultiGoal(
            maze=FOUR_ROOM_MAZE,
            episode_len=60,
            start_pos='random',
            use_stay_action=True,
            reward_type=reward_type,
            goal_pos=FOUR_ROOM_MULTIGOAL_POS,
            goal_rewards=FOUR_ROOM_MULTIGOAL_REWARDS,
            end_at_goal=end_at_goal)
        super(FourRoomMultiGoalRandomStart, self).__init__(task)



class OneRoomRandomGoalRandomStart(env_base.Environment):
    def __init__(self, 
            room_size=5, 
            reward_type='neg', 
            end_at_goal=False):
        task = maze2d_single_goal.Maze2DSingleGoal(
            maze=maze.Maze(maze.SquareRoomFactory(size=room_size)),
            episode_len=room_size*3,
            start_pos='random',
            use_stay_action=True,
            reward_type=reward_type,
            goal_pos=None,
            end_at_goal=end_at_goal)
        super(OneRoomRandomGoalRandomStart, self).__init__(task)


class TwoRoomRandomGoalRandomStart(env_base.Environment):
    def __init__(self,
            room_size=5,
            reward_type='neg', 
            end_at_goal=False):
        task = maze2d_single_goal.Maze2DSingleGoal(
            maze=maze.Maze(maze.TwoRoomsFactory(size=room_size)),
            episode_len=room_size*3,
            start_pos='random',
            use_stay_action=True,
            reward_type=reward_type,
            goal_pos=None,
            end_at_goal=end_at_goal)
        super(TwoRoomRandomGoalRandomStart, self).__init__(task)


class FourRoomRandomGoalRandomStart(env_base.Environment):
    def __init__(self,
            room_size=9,
            reward_type='neg', 
            end_at_goal=False):
        task = maze2d_single_goal.Maze2DSingleGoal(
            maze=maze.Maze(maze.FourRoomsFactory(size=room_size)),
            episode_len=room_size*6,
            start_pos='random',
            use_stay_action=True,
            reward_type=reward_type,
            goal_pos=None,
            end_at_goal=end_at_goal)
        super(FourRoomRandomGoalRandomStart, self).__init__(task)
