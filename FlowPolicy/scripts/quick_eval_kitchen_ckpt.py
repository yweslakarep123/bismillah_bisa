#!/usr/bin/env python3
"""Quick eval of a kitchen checkpoint without Hydra runtime."""

import argparse
import pathlib
import sys

import dill
import hydra
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from train import TrainFlowPolicyWorkspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.checkpoint).resolve()
    out_dir = ckpt_path.parent.parent

    payload = torch.load(
        ckpt_path.open("rb"), pickle_module=dill, map_location="cpu"
    )
    cfg = payload["cfg"]
    ws = TrainFlowPolicyWorkspace(cfg, output_dir=str(out_dir))
    ws.load_payload(payload, exclude_keys=[], include_keys=["model"])
    policy = ws.model.cuda()
    policy.eval()

    runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=str(out_dir),
        eval_episodes=args.eval_episodes,
    )
    log = runner.run(policy)
    print("=== EVAL RESULTS ===")
    for key in sorted(log.keys()):
        if any(token in key for token in ("success", "score", "latency")):
            print(f"{key}: {log[key]}")


if __name__ == "__main__":
    main()
