from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype,
    )


def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x[-n:])
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method="max"):
    if isinstance(data[0], torch.Tensor):
        if method == "max":
            return torch.max(torch.stack(data))
        if method == "min":
            return torch.min(torch.stack(data))
        if method == "mean":
            return torch.mean(torch.stack(data))
        if method == "sum":
            return torch.sum(torch.stack(data))
        raise NotImplementedError()
    if method == "max":
        return np.max(data)
    if method == "min":
        return np.min(data)
    if method == "mean":
        return np.mean(data)
    if method == "sum":
        return np.sum(data)
    raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros(
            (n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype
        )
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros(
            (n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype
        )
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            result[:start_idx] = result[start_idx]
    else:
        raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
    return result


class GymnasiumMultiStepWrapper(gym.Wrapper):
    """Multi-step wrapper for Gymnasium API (5-tuple step)."""

    def __init__(
        self,
        env,
        n_obs_steps,
        n_action_steps,
        max_episode_steps=None,
        reward_agg_method="sum",
    ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(
            env.observation_space, n_obs_steps
        )
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method

        self.obs = deque(maxlen=n_obs_steps + 1)
        self.reward = list()
        self.terminated = list()
        self.truncated = list()
        self.info = defaultdict(lambda: deque(maxlen=n_obs_steps + 1))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs = deque([obs], maxlen=self.n_obs_steps + 1)
        self.reward = list()
        self.terminated = list()
        self.truncated = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))
        return self._get_obs(self.n_obs_steps), info

    def step(self, action):
        for act in action:
            if len(self.terminated) > 0 and self.terminated[-1]:
                break
            if len(self.truncated) > 0 and self.truncated[-1]:
                break
            observation, reward, terminated, truncated, info = self.env.step(act)
            self.obs.append(observation)
            self.reward.append(reward)
            if (
                self.max_episode_steps is not None
                and len(self.reward) >= self.max_episode_steps
            ):
                truncated = True
            self.terminated.append(terminated)
            self.truncated.append(truncated)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        terminated = aggregate(self.terminated, "max")
        truncated = aggregate(self.truncated, "max")
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, terminated, truncated, info

    def _get_obs(self, n_steps=1):
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        if isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs], n_steps
                )
            return result
        raise RuntimeError("Unsupported space type")

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
