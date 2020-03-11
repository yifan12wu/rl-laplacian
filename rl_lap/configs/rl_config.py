from ..agent import dqn_agent
from ..tools import flag_tools

Flags = flag_tools.Flags


class RlConfig(flag_tools.ConfigBase):

    def _set_default_flags(self):
        flags = self._flags
        flags.device = None
        # agent
        flags.discount = 0.99
        flags.update_freq = 1
        flags.update_rate = 0.001
        flags.batch_size = 128
        flags.actor_args = 0.2 # eps greedy
        # optimizer, default is a single opt
        flags.optimizers = Flags()
        flags.optimizers.optimizer = Flags(
                name='Adam', lr=0.001)
        # train
        flags.log_dir = '/tmp/rl/log'
        flags.total_train_steps = int(1e5)
        flags.print_freq = 2000
        flags.test_freq = 5000
        flags.save_freq = 50000
        # replay buffer
        flags.replay_buffer_init = 10000
        flags.replay_buffer_size = int(1e6)
        flags.replay_update_freq = 1
        flags.replay_update_num = 1
        # test
        flags.n_test_episodes = 50


    def _build(self):
        self._build_env()
        self._build_modules()
        self._build_agent_args()

    def _build_env(self):
        dummy_env = self._train_env_factory()
        dummy_time_step = dummy_env.reset()
        self._action_spec = dummy_env.action_spec
        self._obs_shape = list(self._obs_prepro(
            dummy_time_step.observation).shape)

    def _train_env_factory(self):
        raise NotImplementedError

    def _test_env_factory(self):
        raise NotImplementedError

    def _obs_prepro(self, obs):
        raise NotImplementedError

    def _build_modules(self):
        raise NotImplementedError

    def _build_agent_args(self):
        args = Flags()
        args.device = self._flags.device
        # env args
        args.obs_shape = self._obs_shape
        args.obs_prepro = self._obs_prepro
        args.action_spec = self._action_spec
        args.train_env_factory = self._train_env_factory
        args.test_env_factory = self._test_env_factory
        # agent args
        args.modules = self._modules
        args.optimizers = self._flags.optimizers
        args.batch_size = self._flags.batch_size
        args.discount = self._flags.discount
        args.update_freq = self._flags.update_freq
        args.update_rate = self._flags.update_rate
        args.actor_args = self._flags.actor_args
        # training args
        args.log_dir = self._flags.log_dir
        args.total_train_steps = self._flags.total_train_steps
        args.print_freq = self._flags.print_freq
        args.test_freq = self._flags.test_freq
        args.save_freq = self._flags.save_freq
        args.n_test_episodes = self._flags.n_test_episodes
        args.replay_buffer_size = self._flags.replay_buffer_size
        args.replay_buffer_init = self._flags.replay_buffer_init
        args.replay_update_freq = self._flags.replay_update_freq
        args.replay_update_num = self._flags.replay_update_num
        self._agent_args = args

    @property
    def agent_args(self):
        return vars(self._agent_args)



class DqnConfig(RlConfig):

    def _q_module_factory(self):
        raise NotImplementedError

    def _build_modules(self):
        q_module = self._q_module_factory()
        self._modules = dqn_agent.DqnModules(
                q_module, self._obs_shape, self._action_spec)

