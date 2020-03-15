import os
import logging
import collections
import time

import numpy as np
import torch
from torch import optim

from . import replay_buffer
from ..envs import actors
from ..envs import evaluation
from ..envs import env_base
from ..tools import pytools

class EmptyClass(object):
    pass


class BaseAgent(object):

    @pytools.store_args
    def __init__(self,
            # env args
            obs_shape=None,
            obs_prepro=None,
            action_spec=None,
            train_env_factory=None,
            test_env_factory=None,
            # model args
            modules=None, # TODO: replace with module_args
            optimizers=None,
            batch_size=128,
            discount=0.99,
            update_freq=1,
            update_rate=0.001,
            # actor args
            actor_args=0.1,  # e.g. exploration
            replay_buffer_init=10000,
            replay_buffer_size=int(1e6),
            replay_update_freq=1,
            replay_update_num=1,
            # training args
            log_dir='/tmp/rl/log',
            total_train_steps=50000,
            print_freq=1000,
            test_freq=5000,
            save_freq=5000,
            n_test_episodes=50,
            # pytorch
            device=None,
            ):
        self._build_agent()


    def _build_agent(self):
        if self._device is None:
            self._device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        logging.info('device: {}.'.format(self._device))
        self._build_model_fns()
        self._build_optimizers()
        self._replay_buffer = replay_buffer.replaybuffer(
                max_size=self._replay_buffer_size)
        self._global_step = 0
        self._train_info = collections.ordereddict()

    def _build_optimizers(self): 
        # default e.g. for dqn
        opt = self._optimizers.optimizer
        opt_fn = getattr(optim, opt.name)
        self._optimizer = opt_fn(
                self._learning_fns.vars_train,
                lr=opt.lr,
                )

    def _build_model_fns(self):
        raise NotImplementedError

    def _build_loss(self, batch):
        raise NotImplementedError
    
    def _update_target_fns(self):
        vars_learning = self._learning_fns.vars_sync
        vars_target = self._target_fns.vars_sync
        for var_name, var_t in vars_target.items():
            updated_val = (self._update_rate
                    * vars_learning[var_name].data
                    + (1.0 - self._update_rate) * var_t.data)
            var_t.data.copy_(updated_val)

    def _train_policy_fn(self, state):
        raise NotImplementedError
        # return action, context

    def _test_policy_fn(self, state):
        raise NotImplementedError
        # return action, context

    def _random_policy_fn(self, state):
        return self._action_spec.sample(), None

    def _get_obs_batch(self, steps):
        # each step is a tuple of (time_step, action, context)
        obs_batch = [self._obs_prepro(ts[0].observation) for ts in steps]
        return np.stack(obs_batch, axis=0)

    def _get_action_batch(self, steps):
        action_batch = [ts[1] for ts in steps]
        return np.stack(action_batch, axis=0)

    def _get_r_dsc_batch(self, ts2):
        """
        Compute discount based on s_t+1, discount is 0 when
        s_t+1 is the final state, self._discount otherwise.
        """
        at_goal_s2 = np.array(
            [ts[0].step_type==env_base.StepType.FINAL for ts in ts2]
            ).astype(np.float32)
        dsc = (1.0 - at_goal_s2) * self._discount
        r = np.array([ts[0].reward for ts in ts2])
        return r, dsc

    def _tensor(self, x):
        # return a torch.Tensor, assume x is a np array
        if x.dtype in [np.float32, np.float64]:
            return torch.tensor(x, dtype=torch.float32, device=self._device)
        elif x.dtype in [np.int32, np.int64, np.uint8]:
            return torch.tensor(x, dtype=torch.int64, device=self._device)
        else:
            raise ValueError('Unknown dtype {}.'.format(str(x.dtype)))

    def _get_train_batch(self):
        ts1, ts2 = self._replay_buffer.sample_transitions(
            batch_size=self._batch_size,
            )
        a = self._get_action_batch(ts1)
        s1, s2 = map(self._get_obs_batch, [ts1, ts2])
        # compute reward and discount
        r, dsc = self._get_r_dsc_batch(ts2)
        batch = EmptyClass()
        batch.s1 = self._tensor(s1)
        batch.s2 = self._tensor(s2)
        batch.r = self._tensor(r)
        batch.dsc = self._tensor(dsc)
        batch.a = self._tensor(a)
        return batch

    def _optimize_step(self, batch):
        loss = self._build_loss(batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _train_step(self):
        train_batch = self._get_train_batch()
        self._optimize_step(train_batch)
        self._global_step += 1
        if self._global_step % self._update_freq == 0:
            self._update_target_fns()


    def _print_train_info(self):
        info = self._train_info
        summary_str = 'Step {}; '.format(self._global_step)
        for key, val in info.items():
            if isinstance(val, int):
                summary_str += '{} {}; '.format(key, val)
            else:
                summary_str += '{} {:.4g}; '.format(key, val)
        logging.info(summary_str)

    def _save_steps(self, steps):
        self._replay_buffer.add_steps(steps)

    def train(self):
        saver_dir = self._log_dir
        if not os.path.exists(saver_dir):
            os.makedirs(saver_dir)
        #test_agent = TestAgent(learning_agent)
        actor = actors.SingleActor(self._train_env_factory)
        evaluator = evaluation.BasicEvaluator(self._test_env_factory)
        result_path = os.path.join(saver_dir, 'result.csv')

        # start actors, collect trajectories from random actions
        logging.info('Start collecting transitions.')
        start_time = time.time()
        # collect initial transitions
        total_n_steps = 0
        collect_batch = 10000
        while total_n_steps < self._replay_buffer_init:
            n_steps = min(collect_batch, 
                    self._replay_buffer_init - total_n_steps)
            steps = actor.get_steps(n_steps, self._random_policy_fn)
            self._save_steps(steps)
            total_n_steps += n_steps
            logging.info('({}/{}) steps collected.'
                .format(total_n_steps, self._replay_buffer_init))
        time_cost = time.time() - start_time
        logging.info('Replay buffer initialization finished, time cost: {}s'
            .format(time_cost))
        # learning begins
        start_time = time.time()
        test_results = []
        for step in range(self._total_train_steps):
            assert step == self._global_step
            self._train_step()
            # update replay memory:
            if (step + 1) % self._replay_update_freq == 0:
                steps = actor.get_steps(self._replay_update_num,
                        self._train_policy_fn)
                self._save_steps(steps)
            # save
            if (step + 1) % self._save_freq == 0:
                saver_path = os.path.join(saver_dir, 
                        'agent-{}.ckpt'.format(step+1))
                self.save_ckpt(saver_path)
            # print info
            if step == 0 or (step + 1) % self._print_freq == 0:
                time_cost = time.time() - start_time
                logging.info('Training steps per second: {:.4g}.'
                        .format(self._print_freq/time_cost))
                self._print_train_info()
                start_time = time.time()
            # test
            if step == 0 or (step + 1) % self._test_freq == 0:
                tst = time.time()
                test_result = evaluator.run_test(self._n_test_episodes,
                        self._test_policy_fn)
                edt = time.time()
                test_results.append(
                        [step+1] + list(test_result) + [edt-tst])
                self._print_test_info(test_results)
        saver_path = os.path.join(saver_dir, 'agent.ckpt')
        self.save_ckpt(saver_path)
        test_results = np.array(test_results)
        np.savetxt(result_path, test_results, fmt='%.4g', delimiter=',')

    def _print_test_info(self, results):
        if len(results) > 0:
            res = results[-1]
            logging.info(
                    'Tested {} episodes at step {}, '
                    'reward mean {:.4g}, std {:.4g}, time cost {:.4g}s.'
                     .format(self._n_test_episodes, res[0],
                         res[1], res[2], res[3]))

    def save_ckpt(self, filepath):
        torch.save(self._learning_fns.vars_save, filepath)


class BaseModules(object):

    def __init__(self, modules, obs_shape, action_spec):
        self._modules = modules
        self._obs_shape = obs_shape
        self._action_spec = action_spec

    def build(self, model='learn', device=torch.device('cpu')):
        # should return an object that contains (e.g.):
        # vars_save, vars_train, vars_sync
        raise NotImplementedError









