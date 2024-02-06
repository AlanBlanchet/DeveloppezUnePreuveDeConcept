import gymnasium
from gymnasium.spaces import Discrete


class Environment:
    def __init__(self, env_id: str):
        self.env_id = env_id
        self.setup()

    def setup(self):
        self._env = gymnasium.make(self.env_id, render_mode="rgb_array")

    def reset(self):
        return self._env.reset()

    def clean(self):
        self.reset()
        self._env.close()

    def step(self, action):
        return self._env.step(action)

    def set_render(self, render_mode: str):
        self._env.close()
        self._env = gymnasium.make(self._env.spec.id, render_mode=render_mode)
        return self

    @property
    def render_mode(self):
        return self._env.render_mode

    @property
    def state_shape(self):
        if isinstance(self._env.observation_space, Discrete):
            return self._env.observation_space.n
        else:
            return self._env.observation_space.shape

    @property
    def out_action(self):
        if isinstance(self._env.action_space, Discrete):
            return self._env.action_space.n
        else:
            return self._env.action_space.shape

    @property
    def action_names(self):
        unwrapped = self._env.unwrapped
        if hasattr(unwrapped, "get_action_meanings"):
            return unwrapped.get_action_meanings()
        return range(self.out_action)

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        del state["_env"]
        return state
