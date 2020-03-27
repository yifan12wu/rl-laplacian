'''
Visualize learned representation.
Read checkpoint, config, environment.
get all representations

'''
import os
import argparse

from rl_lap.tools import flag_tools
from rl_lap.tools import logging_tools


parser = argparse.ArgumentParser()
parser.add_argument('--log_base_dir', type=str, 
        default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--log_sub_dir', type=str, default='test')


FLAGS = parser.parse_args()


def get_config_cls():
    config_module = importlib.import_module(
            FLAGS.config_dir+'.'+FLAGS.config_file)
    config_cls = config_module.Config
    return config_cls


def main():
    # pass args to config
    cfg_cls = get_config_cls()
    flags = flag_tools.Flags()
    flags.log_dir = os.path.join(
            FLAGS.log_base_dir,
            FLAGS.exp_name,
            FLAGS.env_id,
            FLAGS.log_sub_dir)
    flags.env_id = FLAGS.env_id
    flags.args = FLAGS.args
    logging_tools.config_logging(flags.log_dir)
    cfg = cfg_cls(flags)
    flag_tools.save_flags(cfg.flags, flags.log_dir)
    learner = laprepr.LapReprLearner(**cfg.args)
    learner.train()


if __name__ == '__main__':
    main()


import os
import numpy as np
import tensorflow.google as tf

#import matplotlib
#matplotlib.use('Agg')

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from google3.pyglib import gfile

import seaborn
seaborn.reset_orig()

params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)


class FLAGS(object):
  n_test_episodes = 50
  env_id = 'HardMaze2'
  goal_pos = (10, 10)  # OneRoom: 15, 15, TwoRoom: 9, 15, HardMaze2: 10, 10
  repr_dim = 20
  dist_params = [2, 2]
  reward_type = 'neg'
  
config.setup(FLAGS)
env = config.env_factory()

n_states = env.task.maze.n_states
# take a dummy action to get the first state of the env
dummy_time_step = env.reset()
image_shape = list(dummy_time_step.observation.image.shape)
obs_shape = list(config.obs_to_repr(dummy_time_step.observation).shape)

pos_batch = env.task.maze.all_empty_grids()
goal_obs = env.task.get_goal_observation()
goal_state = np.expand_dims(config.obs_to_repr(goal_obs), axis=0)
#goal_state[0, 0] = goal_state[0, 1]
obs_batch = [env.task.pos_to_obs(pos_batch[i]) for i in range(pos_batch.shape[0])]
states_batch = np.array([config.obs_to_repr(obs) for obs in obs_batch])

print(states_batch.shape, goal_state.shape)
print(goal_state)

exp_id = 1807231053
# env_name = 'HardMaze2'
log_base_dir = '/cns/li-d/home/wuyifan/exp/log/rewardshaping/grid/{}'.format(exp_id)
log_sub_dir = 'p{}_shaped_mix_0'.format(FLAGS.env_id)
log_dir = os.path.join(log_base_dir, log_sub_dir)
fig_dir = os.path.join(log_base_dir, 'figs')
fig_path = os.path.join(fig_dir, FLAGS.env_id+'_repr_rs.pdf')
fig_l2_path = os.path.join('/usr/local/google/home/wuyifan/data/figs/paper', FLAGS.env_id+'_dist_l2.png')
#log_dir = '/cns/is-d/home/wuyifan/log/rewardshaping/tmp'
#log_base_dir = '/cns/is-d/home/wuyifan/log/rewardshaping/tmp'
#log_dir = os.path.join(log_base_dir, FLAGS.env_id)
model_path = os.path.join(log_dir, 'pretrain')

def dist_fn(x, p, q):
  '''return (sum_i |x_i|^p)^{1/q} for each row.'''
  return tf.pow(tf.reduce_sum(tf.pow(x, p), axis=-1), 1/q)

tf.reset_default_graph()
with tf.variable_scope('repr_net'):
  repr_net = config.ReprNet(repr_dim=FLAGS.repr_dim)
tf_all_states = tf.constant(states_batch, tf.float32)
tf_goal_state = tf.constant(goal_state, tf.float32)
tf_repr_s = repr_net(tf_all_states)
tf_repr_g = repr_net(tf_goal_state)
ns = tf_repr_s.shape.as_list()[0]
tf_repr_g_tiled = tf.tile(tf_repr_g, [ns, 1])
tf_g_tiled = tf.tile(tf_goal_state, [ns, 1])

tf_dists = dist_fn(
    tf_repr_s - tf_repr_g_tiled,
    FLAGS.dist_params[0], 
    FLAGS.dist_params[1])

tf_dists_l2 = dist_fn(
    tf_all_states - tf_g_tiled,
    FLAGS.dist_params[0], 
    FLAGS.dist_params[1])


tf.contrib.framework.init_from_checkpoint(model_path, {'w2v_learner/repr_net/': 'repr_net/'})

with tf_utils.tf_session() as sess:
  dists, dists_l2 = sess.run([tf_dists, tf_dists_l2])


map_ = np.zeros(image_shape[:2], dtype=np.float32)
map_[pos_batch[:, 0], pos_batch[:, 1]] = dists_l2
im_ = plt.imshow(map_, interpolation='none', cmap='Blues')
plt.colorbar()

vmin = np.amin(map_)
vmax = np.amax(map_)

walls = np.tile(np.expand_dims(env.task.maze.render(), axis=-1), [1, 1, 4])
#map_2 = np.zeros([image_shape[0], image_shape[1], 4])
#map_2[:, :, 2] = - (map_ - vmin) / (vmax - vmin) 
map_2 = im_.cmap(im_.norm(map_))
map_2 += walls * 0.5
map_2[FLAGS.goal_pos] = [1, 0, 0, 1]
plt.imshow(map_2, interpolation='none')

with gfile.Open(fig_l2_path, 'w') as fout:
  plt.savefig(fout, bbox_inches='tight')

plt.show()

plt.clf()