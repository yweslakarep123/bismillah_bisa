#!/usr/bin/env python3
"""Convert Minari D4RL/kitchen/complete-v2 to zarr for FlowPolicy training."""

import argparse
import json
import os
from pathlib import Path

import minari
import numpy as np
import zarr
from termcolor import cprint


def extract_episode(episode):
    obs = np.asarray(episode.observations["observation"], dtype=np.float32)
    act = np.asarray(episode.actions, dtype=np.float32)
    # align off-by-one: actions are one step shorter than observations
    obs = obs[: len(act)]
    return obs, act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="D4RL/kitchen/complete-v2",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for zarr and splits json (default: FlowPolicy/data)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    data_dir = Path(args.output_dir) if args.output_dir else package_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = data_dir / "kitchen_complete.zarr"
    splits_path = data_dir / "kitchen_complete_splits.json"

    dataset = minari.load_dataset(args.dataset_id)
    n_episodes = dataset.total_episodes
    cprint(f"Loaded {args.dataset_id}: {n_episodes} episodes", "green")

    if n_episodes != 19:
        cprint(f"Warning: expected 19 episodes, got {n_episodes}", "yellow")

    # fixed split: 15 train / 3 val / 1 test (episode-level)
    train_ids = list(range(15))
    val_ids = list(range(15, 18))
    test_ids = [18] if n_episodes > 18 else [n_episodes - 1]

    state_arrays = []
    action_arrays = []
    episode_ends = []
    total_count = 0

    for ep_idx in range(n_episodes):
        ep = dataset[ep_idx]
        obs, act = extract_episode(ep)
        state_arrays.append(obs)
        action_arrays.append(act)
        total_count += len(act)
        episode_ends.append(total_count)
        cprint(
            f"Episode {ep_idx}: {len(act)} steps, obs_dim={obs.shape[1]}",
            "cyan",
        )

    state_arrays = np.concatenate(state_arrays, axis=0)
    action_arrays = np.concatenate(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends, dtype=np.int64)

    splits = {
        "seed": args.seed,
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
        "test_episode_ids": test_ids,
        "n_episodes": n_episodes,
    }
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    if zarr_path.exists():
        import shutil
        shutil.rmtree(zarr_path)

    zarr_root = zarr.group(str(zarr_path))
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    state_chunk_size = (100, state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])

    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=action_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    cprint(f"state shape: {state_arrays.shape}", "green")
    cprint(f"action shape: {action_arrays.shape}", "green")
    cprint(f"episode_ends: {episode_ends_arrays}", "green")
    cprint(f"Saved zarr to {zarr_path}", "green")
    cprint(f"Saved splits to {splits_path}", "green")


if __name__ == "__main__":
    main()
