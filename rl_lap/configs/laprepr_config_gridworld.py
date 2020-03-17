from ..agent import laprepr
from ..envs.gridworld import gridworld_envs
from . import networks


class Config(laprepr.LapReprConfig):

    def _obs_prepro(self, obs):
        return obs.agent.position

    def _env_factory(self):
        return gridworld_envs.make(self._flags.env_id)

    def _model_factory(self):
        return networks.ReprNetMLP(
                self._obs_shape, n_layers=2, n_units=200,
                d=self._flags.model_args.d)


