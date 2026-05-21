import copy
import json
from typing import Dict, List

import numpy as np
import torch

from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.common.replay_buffer import ReplayBuffer
from flow_policy_3d.common.sampler import SequenceSampler
from flow_policy_3d.dataset.base_dataset import BaseDataset
from flow_policy_3d.model.common.normalizer import LinearNormalizer


def load_episode_mask(
    episode_mask_path: str,
    n_episodes: int,
    split: str,
) -> np.ndarray:
    with open(episode_mask_path, "r") as f:
        splits = json.load(f)
    if split == "train":
        ids: List[int] = splits["train_episode_ids"]
    elif split == "val":
        ids = splits["val_episode_ids"]
    elif split == "test":
        ids = splits["test_episode_ids"]
    else:
        raise ValueError(f"Unknown split: {split}")
    mask = np.zeros(n_episodes, dtype=bool)
    for i in ids:
        if i < n_episodes:
            mask[i] = True
    return mask


class KitchenDataset(BaseDataset):
    def __init__(
        self,
        zarr_path: str,
        episode_mask_path: str,
        horizon: int = 8,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        split: str = "train",
        augment_obs: bool = False,
        obs_noise_std: float = 0.01,
    ):
        super().__init__()
        self.episode_mask_path = episode_mask_path
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["state", "action"]
        )
        n_episodes = self.replay_buffer.n_episodes
        episode_mask = load_episode_mask(episode_mask_path, n_episodes, split)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask,
        )
        self.episode_mask = episode_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.split = split
        self.augment_obs = augment_obs and split == "train"
        self.obs_noise_std = obs_noise_std
        self._rng = np.random.default_rng(seed)

    def get_validation_dataset(self) -> "KitchenDataset":
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=load_episode_mask(
                self.episode_mask_path,
                self.replay_buffer.n_episodes,
                "val",
            ),
        )
        val_set.episode_mask = load_episode_mask(
            self.episode_mask_path,
            self.replay_buffer.n_episodes,
            "val",
        )
        val_set.split = "val"
        val_set.augment_obs = False
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)
        if self.augment_obs and self.obs_noise_std > 0:
            noise = self._rng.normal(
                0, self.obs_noise_std, size=agent_pos.shape
            ).astype(np.float32)
            agent_pos = agent_pos + noise
        return {
            "obs": {"agent_pos": agent_pos},
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
