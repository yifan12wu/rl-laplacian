from ..agent import lap_repr
from ..envs.gridworld import gridworld_envs
from . import networks


class Config(lap_repr.LapReprConfig):

    def _env_factory(self):
        return gridworld_envs.make(self._flags.env_id)

    def _q_module_factory(self):
        return networks.ReprNetMLP(
                self._obs_shape, n_layers=2, n_units=200,
                d=self._flags.model_args.d)


