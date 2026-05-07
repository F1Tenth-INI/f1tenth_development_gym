from sim.f110_sim.envs.pygame_rendering import EnvRenderer as _PygameEnvRenderer


class EnvRenderer:
    """Adapter exposing the renderer API expected by run_simulation."""

    def __init__(self, width, height):
        self._impl = _PygameEnvRenderer(width, height)

    def update_map(self, map_path, map_ext):
        self._impl.update_map(map_path, map_ext)

    def render(self, obs):
        self._impl.update_obs(obs)
        self._impl.render()

    def close(self):
        self._impl.close()

