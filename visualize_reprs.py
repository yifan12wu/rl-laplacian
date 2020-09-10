"""Visualize learned representation."""
import os
import argparse
import importlib

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

from rl_lap.agent import laprepr
from rl_lap.tools import flag_tools
from rl_lap.tools import torch_tools


parser = argparse.ArgumentParser()
parser.add_argument('--log_base_dir', type=str, 
        default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--log_sub_dir', type=str, 
        default='laprepr/OneRoom/test')
parser.add_argument('--output_sub_dir', type=str, 
        default='visualize_reprs')
parser.add_argument('--config_dir', type=str, default='rl_lap.configs')
parser.add_argument('--config_file', 
        type=str, default='laprepr_config_gridworld')


FLAGS = parser.parse_args()


def get_config_cls():
    config_module = importlib.import_module(
            FLAGS.config_dir+'.'+FLAGS.config_file)
    config_cls = config_module.Config
    return config_cls


def main():
    # setup log directories
    log_dir = os.path.join(FLAGS.log_base_dir, FLAGS.log_sub_dir)
    output_dir = os.path.join(FLAGS.log_base_dir, FLAGS.output_sub_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # load config
    flags = flag_tools.load_flags(log_dir)
    cfg_cls = get_config_cls()
    cfg = cfg_cls(flags)
    learner_args = cfg.args_as_flags
    device = learner_args.device
    # load model from checkpoint
    model = learner_args.model_cfg.model_factory()
    model.to(device=device)
    ckpt_path = os.path.join(log_dir, 'model.ckpt')
    model.load_state_dict(torch.load(ckpt_path))
    # -- use loaded model to get state representations --
    # get the full batch of states from env
    env = learner_args.env_factory()
    obs_prepro = learner_args.obs_prepro
    n_states = env.task.maze.n_states
    pos_batch = env.task.maze.all_empty_grids()
    obs_batch = [env.task.pos_to_obs(pos_batch[i]) for i in range(n_states)]
    states_batch = np.array([obs_prepro(obs) for obs in obs_batch])
    # get goal state representation
    goal_pos = env.task.goal_pos
    goal_obs = env.task.pos_to_obs(goal_pos)
    goal_state = obs_prepro(goal_obs)[None]
    # get representations from loaded model
    states_torch = torch_tools.to_tensor(states_batch, device)
    goal_torch = torch_tools.to_tensor(goal_state, device)
    states_reprs = model(states_torch).detach().cpu().numpy()
    goal_repr = model(goal_torch).detach().cpu().numpy()
    # compute l2 distances from states to goal
    l2_dists = np.sqrt(np.sum(np.square(states_reprs - goal_repr), axis=-1))
    # -- visialize state representations --
    # plot raw distances with the walls
    image_shape = goal_obs.agent.image.shape
    map_ = np.zeros(image_shape[:2], dtype=np.float32)
    map_[pos_batch[:, 0], pos_batch[:, 1]] = l2_dists
    im_ = plt.imshow(map_, interpolation='none', cmap='Blues')
    plt.colorbar()
    # add the walls to the normalized distance plot
    walls = np.expand_dims(env.task.maze.render(), axis=-1)
    map_2 = im_.cmap(im_.norm(map_))
    map_2[:, :, :-1] = map_2[:, :, :-1] * (1 - walls) + 0.5 * walls
    map_2[:, :, -1:] = map_2[:, :, -1:] * (1 - walls) + 1.0 * walls
    map_2[goal_pos[0], goal_pos[1]] = [1, 0, 0, 1]
    plt.cla()
    plt.imshow(map_2, interpolation='none')
    plt.xticks([])
    plt.yticks([])
    figfile = os.path.join(output_dir, '{}.pdf'.format(flags.env_id))
    plt.savefig(figfile, bbox_inches='tight')
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    main()

