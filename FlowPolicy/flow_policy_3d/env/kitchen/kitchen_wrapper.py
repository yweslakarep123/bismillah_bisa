import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)


class KitchenEnvWrapper(gym.Wrapper):
    """Wrap FrankaKitchen-v1: expose flat state as agent_pos for FlowPolicy."""

    def __init__(
        self,
        tasks_to_complete=None,
        max_episode_steps=280,
    ):
        if tasks_to_complete is None:
            tasks_to_complete = [
                "microwave",
                "kettle",
                "light switch",
                "slide cabinet",
            ]
        env = gym.make(
            "FrankaKitchen-v1",
            tasks_to_complete=tasks_to_complete,
            max_episode_steps=max_episode_steps,
        )
        super().__init__(env)
        self.tasks_to_complete = tasks_to_complete
        obs_dim = env.observation_space["observation"].shape[0]
        self._observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self._action_space = env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _wrap_obs(self, obs):
        return {
            "agent_pos": np.asarray(obs["observation"], dtype=np.float32),
        }

    def _wrap_info(self, info):
        if info is None:
            info = {}
        info = dict(info)
        completions = info.get("episode_task_completions", [])
        n_completed = len(completions) if completions is not None else 0
        info["num_tasks_completed"] = n_completed
        info["goal_achieved"] = np.array(
            [n_completed >= len(self.tasks_to_complete)], dtype=bool
        )
        # per-subtask success rates (k=1..4)
        for k in range(1, len(self.tasks_to_complete) + 1):
            info[f"success_k{k}"] = n_completed >= k
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._wrap_obs(obs), self._wrap_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._wrap_obs(obs),
            reward,
            terminated,
            truncated,
            self._wrap_info(info),
        )
