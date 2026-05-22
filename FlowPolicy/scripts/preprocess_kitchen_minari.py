#!/usr/bin/env python3
"""Convert Minari D4RL/kitchen/complete-v2 to zarr for FlowPolicy training.

Follows the DAgger4Robotics preprocessing pattern:
https://github.com/cybernetic-m/DAgger4Robotics/blob/main/utils/preprocess_dataset.py

Extended from a single microwave task to four sequential tasks:
microwave -> kettle -> light switch -> slide cabinet.
"""

import argparse
import json
from pathlib import Path

import minari
import numpy as np
import zarr
from termcolor import cprint

# Task order in the expert complete-v2 demonstrations.
TASK_ORDER = ["microwave", "kettle", "light switch", "slide cabinet"]

# DAgger4Robotics uses this tolerance for scalar achieved-goal stability.
STABILITY_TOLERANCE = 1e-20

# FrankaKitchen env completion threshold (kitchen_env.BONUS_THRESH).
BONUS_THRESH = 0.3


def _is_stable(achieved, step_index: int) -> bool:
    """True when achieved goal stops changing between step_index and step_index+1."""
    ag = np.asarray(achieved)
    if ag.ndim == 1:
        return abs(ag[step_index + 1] - ag[step_index]) < STABILITY_TOLERANCE
    return np.max(np.abs(ag[step_index + 1] - ag[step_index])) < STABILITY_TOLERANCE


def _is_near_goal(achieved, desired, step_index: int) -> bool:
    """True when achieved goal is within FrankaKitchen completion distance."""
    ag_i = np.asarray(achieved[step_index]).flatten()
    desired_arr = np.asarray(desired)
    if desired_arr.ndim > 1:
        dg_i = desired_arr[step_index].flatten()
    else:
        dg_i = desired_arr.flatten()
    return float(np.linalg.norm(ag_i - dg_i)) < BONUS_THRESH


def _is_task_complete(task: str, achieved, desired, step_index: int) -> bool:
    """Detect completion for the current sequential task."""
    if task == "microwave":
        # DAgger4Robotics: microwave completion via achieved-goal stability.
        return _is_stable(achieved, step_index)
    # Vector / later scalar tasks: use the same threshold as FrankaKitchen-v1.
    return _is_near_goal(achieved, desired, step_index)


def extract_episode(episode):
    """Extract obs/action pairs until all four tasks are completed.

    Mirrors the DAgger4Robotics loop: append each step, then stop once the
    final task in TASK_ORDER is finished.
    """
    observation_list = episode.observations["observation"]
    actions_list = episode.actions
    achieved = episode.observations["achieved_goal"]
    desired = episode.observations["desired_goal"]

    obs_chunks = []
    act_chunks = []
    completed_tasks = []

    n_actions = len(actions_list)
    for step_index in range(n_actions - 1):
        obs_chunks.append(
            np.asarray(observation_list[step_index], dtype=np.float32)
        )
        act_chunks.append(np.asarray(actions_list[step_index], dtype=np.float32))

        if len(completed_tasks) < len(TASK_ORDER):
            current_task = TASK_ORDER[len(completed_tasks)]
            if _is_task_complete(
                current_task,
                achieved[current_task],
                desired[current_task],
                step_index,
            ):
                completed_tasks.append(current_task)
                if len(completed_tasks) >= len(TASK_ORDER):
                    break

    if not obs_chunks:
        raise ValueError(
            f"Episode {getattr(episode, 'id', '?')} produced no transitions."
        )

    obs = np.stack(obs_chunks, axis=0)
    act = np.stack(act_chunks, axis=0)
    return obs, act, completed_tasks


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
        obs, act, completed_tasks = extract_episode(ep)
        if len(completed_tasks) != len(TASK_ORDER):
            cprint(
                f"Warning: episode {ep_idx} completed {len(completed_tasks)}/"
                f"{len(TASK_ORDER)} tasks: {completed_tasks}",
                "yellow",
            )

        state_arrays.append(obs)
        action_arrays.append(act)
        total_count += len(act)
        episode_ends.append(total_count)
        cprint(
            f"Episode {ep_idx}: {len(act)} steps (raw {len(ep.actions)}), "
            f"obs_dim={obs.shape[1]}, tasks={completed_tasks}",
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
        "task_order": TASK_ORDER,
        "preprocess": {
            "microwave_completion": "achieved_goal_stability",
            "other_tasks_completion": f"distance_lt_{BONUS_THRESH}",
        },
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
