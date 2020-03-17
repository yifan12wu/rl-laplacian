from ..agent import lap_repr
from . import networks


class Config(lap_repr.LapReprConfig):

    def _env_factory(self):
        pass

    def _q_module_factory(self):
        return networks.ReprNetMLP(
                self._obs_shape, n_layers=2, n_units=200,
                d=self._flags.model_args.d)


