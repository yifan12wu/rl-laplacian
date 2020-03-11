import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ReprNetMLP(nn.Module):

    def __init__(self, obs_shape, action_spec,
            n_layers, n_units):
        super().__init__()
        self._layers = []
        n_in = int(np.prod(np.array(obs_shape)))
        for i in range(n_layers):
            layer = nn.Linear(n_in, n_units)
            self.add_module('hidden_layer_{}'.format(i+1), layer)
            n_in = n_units
            self._layers.append(layer)

    def forward(self, x):
        h = x.reshape(x.shape[0], -1)
        for layer in self._layers:
            h = F.relu(layer(h))
        return h


class DiscreteQNetMLP(nn.Module):

    def __init__(self, obs_shape, action_spec,
            n_layers, n_units, fix_repr=False):
        super().__init__()
        self._fix_repr = fix_repr
        self.repr_fn = ReprNetMLP(obs_shape=obs_shape, 
                action_spec=action_spec, n_layers=n_layers,
                n_units=n_units)
        if n_layers >= 1:
            n_in = n_units
        else:
            n_in = int(np.prod(np.array(obs_shape)))
        self.out_layer = nn.Linear(n_in, action_spec.n)
        # torch.nn.init.uniform_(
        #         self.out_layer.weight, -0.001, 0.001)
        # torch.nn.init.constant_(
        #         self.out_layer.bias, 0.0)

    def forward(self, x):
        h = self.repr_fn(x)
        if self._fix_repr:
            h = h.detach()
        o = self.out_layer(h)
        return o

