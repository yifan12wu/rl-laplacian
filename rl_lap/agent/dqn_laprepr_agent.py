import logging
import collections

import numpy as np
import torch

from . import agent_base
from ..envs import env_base


class EmptyClass(object):
    pass


class DqnAgent(agent_base.BaseAgent):

    def __init__(self, 
            load_model='none',
            load_model_path='',
            **kwargs):
        self._load_model = load_model
        self._load_model_path = load_model_path
        super().__init__(**kwargs)

    def _build_model_fns(self):
        self._learning_fns = self._modules.build(
                mode='learn', device=self._device)
        self._target_fns = self._modules.build(
                mode='target', device=self._device)
        self._q_fn_learning = self._learning_fns.q_fn
        self._q_fn_target = self._target_fns.q_fn
        self._q_fn_target.load_state_dict(
                self._q_fn_learning.state_dict())
        if self._load_model == 'repr':
            state_dict = torch.load(self._load_model_path)
            self._q_fn_learning.repr_fn.load_state_dict(state_dict['repr_fn'])
            self._q_fn_target.repr_fn.load_state_dict(state_dict['repr_fn'])
        elif self._load_model == 'all':
            state_dict = torch.load(self._load_model_path)
            self._q_fn_learning.load_state_dict(state_dict['q_fn'])
            self._q_fn_target.load_state_dict(state_dict['q_fn'])
            

    def _build_loss(self, batch):
        q_loss = self._build_q_loss(batch)
        return q_loss

    def _build_q_loss(self, batch):
        # modules and tensors
        s1 = batch.s1
        s2 = batch.s2
        a = batch.a
        r = batch.r
        dsc = batch.dsc
        batch_size = a.shape[0]
        ######################
        # networks
        q_vals_learning = self._q_fn_learning(s1)
        q_val_learning = q_vals_learning[torch.arange(batch_size), a]
        val_target = self._build_v_target(s2)
        q_val_target = (r + dsc * val_target).detach()
        loss = (q_val_learning - q_val_target).pow(2).mean()
        # build print info
        info = self._train_info
        info['q_loss'] = loss.item()
        info['mean_q'] = q_val_target.mean().item()
        info['min_q'] = q_val_target.min().item()
        info['max_q'] = q_val_target.max().item()
        info['mean_r'] = r.mean().item()
        info['mean_dsc'] = dsc.mean().item()
        # 
        # for i, w in enumerate(self._q_fn_learning.parameters()):
        #     print(i, w.sum().item())
        return loss

    def _build_v_target(self, s):
        q_vals_target = self._q_fn_target(s)
        val_target = q_vals_target.max(-1)[0]
        return val_target


    def _policy_fn(self, state):
        time_step, _ = state
        s = np.expand_dims(self._obs_prepro(time_step.observation), 0)
        s = self._tensor(s)
        with torch.no_grad():
            q_vals = self._q_fn_learning(s).cpu().numpy()
        return q_vals[0]

    def _train_policy_fn(self, state):
        # epsilon greedy
        q_vals = self._policy_fn(state)
        eps = self._actor_args
        if np.random.uniform() <= eps:
            a = self._action_spec.sample()
        else:
            a = np.argmax(q_vals)
        return a, None

    def _test_policy_fn(self, state):
        q_vals = self._policy_fn(state)
        return np.argmax(q_vals), None

    def save_ckpt(self, filepath):
        torch.save(self._learning_fns.vars_save, filepath)
        torch.save(self._target_fns.vars_save, filepath + 'target')


class DqnModules(agent_base.BaseAgent):

    def __init__(self, q_module, obs_shape, action_spec):
        self._q_module = q_module
        self._obs_shape = obs_shape
        self._action_spec = action_spec

    def build(self, mode='learn', device=torch.device('cpu')):
        q_fn = self._q_module(
                self._obs_shape, self._action_spec
                )
        q_fn = q_fn.to(device=device)
        fns = EmptyClass()
        fns.q_fn = q_fn
        fns.vars_save = {
            'q_fn': q_fn.state_dict(), 
            'repr_fn': q_fn.repr_fn.state_dict(),
            'out_layer': q_fn.out_layer.state_dict(),
            }
        fns.vars_sync = q_fn.state_dict()
        fns.vars_train = q_fn.parameters()
        return fns

