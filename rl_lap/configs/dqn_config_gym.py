from . import rl_config
from ..envs import gym_wrapper
from ..agent import q_nets

class Config(rl_config.DqnConfig):

    def _set_default_flags(self):
        super(Config, self)._set_default_flags()
        flags = self._flags
        # new parameters
        flags.env_id = 'CartPole-v0'
        flags.load_model = 'none'  # 'repr', 'none', 'all'
        flags.load_model_path = ''
        flags.fix_repr = False
        # reset some default general parameters
        # agent
        flags.discount = 0.99
        flags.update_freq = 1
        flags.update_rate = 0.005
        flags.batch_size = 128
        flags.actor_args = 0.2 # eps greedy
        flags.optimizers.optimizer.name = 'Adam'
        flags.optimizers.optimizer.lr = 0.001
        # train
        flags.total_train_steps = 150000
        flags.print_freq = 5000
        flags.test_freq = 5000
        flags.save_freq = 50000
        # replay buffer
        flags.replay_buffer_init = 10000
        flags.replay_buffer_size = int(1e6)
        flags.replay_update_freq = 1
        flags.replay_update_num = 1
        # test
        flags.n_test_episodes = 20

    def _train_env_factory(self):
        return gym_wrapper.Environment(self._flags.env_id)

    def _test_env_factory(self):
        return gym_wrapper.Environment(self._flags.env_id)

    def _obs_prepro(self, obs):
        return obs

    def _q_module_factory(self):
        def q_module(obs_shape, action_spec):
            return q_nets.DiscreteQNetMLP(
                    obs_shape, action_spec,
                    n_layers=2, n_units=200,
                    fix_repr=self._flags.fix_repr)
        return q_module

    def _build_agent_args(self):
        super()._build_agent_args()
        args = self._agent_args
        args.load_model = self._flags.load_model
        args.load_model_path = self._flags.load_model_path


