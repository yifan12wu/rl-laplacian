import logging
import numpy as np
import torch
from torch import optim

from . import dqn_agent
from ..tools import flag_tools


DEFAULT_EXP_ARGS = flag_tools.Flags()
DEFAULT_EXP_ARGS.grad_steps = 10
DEFAULT_EXP_ARGS.lr = 0.1
DEFAULT_EXP_ARGS.eps = 0.2
DEFAULT_EXP_ARGS.ref = 'learning'
DEFAULT_EXP_ARGS.sample_replay = True
DEFAULT_EXP_ARGS.beta = 1.0


class DqnRegExpAgent(dqn_agent.DqnAgent):

    def __init__(self, 
            exp_args=DEFAULT_EXP_ARGS,
            **kwargs):
        self._exp_args = exp_args
        logging.info(flag_tools.flags_to_dict(self._exp_args))
        super().__init__(**kwargs)

    def _build_model_fns(self):
        self._learning_fns = self._modules.build(
                mode='learn', device=self._device)
        self._target_fns = self._modules.build(
                mode='target', device=self._device)
        self._exp_fns = self._modules.build(
                mode='explore', device=self._device)
        self._q_fn_learning = self._learning_fns.q_fn
        self._q_fn_target = self._target_fns.q_fn
        self._q_fn_target.load_state_dict(
                self._q_fn_learning.state_dict())
        self._q_fn_exp = self._exp_fns.q_fn

    def _build_optimizers(self):
        super()._build_optimizers()
        # build optimizer for exploration
        self._exp_optimizer = optim.SGD(
                self._q_fn_exp.parameters(),
                lr=self._exp_args.lr) 

    def _train_policy_fn(self, state):
        # epsilon greedy
        eps_reg = self._exp_args.eps
        eps_rand = self._actor_args - eps_reg
        assert eps_rand >= 0.0
        seed = np.random.uniform()
        if seed <= eps_rand:
            a = self._action_spec.sample()
        elif seed <= eps_rand + eps_reg:
            a = self._get_explore_action(state)
        else:
            q_vals = self._policy_fn(state)
            a = np.argmax(q_vals)
        return a, None

    def _get_explore_action(self, state):
        time_step, _ = state
        s = np.expand_dims(self._obs_prepro(time_step.observation), 0)
        s = self._tensor(s)
        if self._exp_args.ref == 'learning':
            q_ref = self._q_fn_learning
        else:
            assert self._exp_args.ref == 'target'
            q_ref = self._q_fn_target
        # get q_ref(s, a) for all a
        with torch.no_grad():
            q_vals_ref = q_ref(s)
        # initialize q_exp with q_ref
        q_exp = self._q_fn_exp
        vars_ref = q_ref.state_dict()
        vars_exp = q_exp.state_dict()
        for var_name, var_t in vars_exp.items():
            var_t.data.copy_(vars_ref[var_name].data)
        # run gradient descent to maximize ||q_ref(s, a) - q_exp(s, a)||^2
        # but this gradient is zero, maximize q_exp(s, a) instead
        # TODO: sample mini-batch to regularize the difference for seen samples
        for _ in range(self._exp_args.grad_steps):
            q_vals_exp = q_exp(s)
            # loss = - (q_vals_exp - q_vals_ref).abs().sum()
            if self._exp_args.sample_replay:
                batch = self._get_train_batch()
                s_rb = batch.s1
                a_rb = batch.a
                batch_size = a_rb.shape[0]
                with torch.no_grad():
                    q_ref_rb = q_ref(s_rb)
                q_exp_rb = q_exp(s_rb)
                q_diff_rb = (q_exp_rb - q_ref_rb)[torch.arange(batch_size), a_rb]
                reg_loss = q_diff_rb.pow(2).mean()
                loss = reg_loss * self._exp_args.beta - q_vals_exp.sum()
            else:
                loss = - q_vals_exp.sum()
            self._exp_optimizer.zero_grad()
            loss.backward()
            self._exp_optimizer.step()
        # select action = argmax_a ||q_ref(s, a) - q_exp(s, a)||^2
        with torch.no_grad():
            q_vals_exp = q_exp(s)
            q_diff = (q_vals_exp - q_vals_ref).abs().cpu().numpy()[0]
        action = np.argmax(q_diff)
        # action = np.argmax(q_vals_exp)
        # print(q_diff, action)
        return action




