from typing import Dict, List, Type

import torch
import torch.nn as nn
from termcolor import cprint

from flow_policy_3d.model.vision.pointnet_extractor import create_mlp


class StateOnlyEncoder(nn.Module):
    """State-only observation encoder for Franka Kitchen (no point cloud)."""

    def __init__(
        self,
        observation_space: Dict,
        out_channel: int = 512,
        state_mlp_size=(512, 512),
        state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
        state_key: str = "agent_pos",
    ):
        super().__init__()
        self.state_key = state_key
        if state_key not in observation_space:
            raise KeyError(
                f"StateOnlyEncoder requires '{state_key}' in observation_space, "
                f"got keys {list(observation_space.keys())}"
            )
        self.state_shape = observation_space[state_key]
        state_dim = self.state_shape[0]

        if len(state_mlp_size) == 0:
            raise RuntimeError("state_mlp_size must not be empty")
        elif len(state_mlp_size) == 1:
            net_arch: List[int] = []
        else:
            net_arch = list(state_mlp_size[:-1])
        output_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(
            *create_mlp(
                state_dim,
                output_dim,
                net_arch,
                state_mlp_activation_fn,
            )
        )
        if output_dim != out_channel:
            self.proj = nn.Linear(output_dim, out_channel)
        else:
            self.proj = nn.Identity()

        self._output_dim = out_channel
        cprint(
            f"[StateOnlyEncoder] state_dim={state_dim}, output_dim={out_channel}",
            "yellow",
        )

    def forward(self, observations: Dict) -> torch.Tensor:
        state = observations[self.state_key]
        feat = self.state_mlp(state)
        return self.proj(feat)

    def output_shape(self):
        return self._output_dim
