#!/usr/bin/env python3
"""
Franka Kitchen experiment pipeline: baseline → random search → multi-seed eval.

Protocol (flowpolicy_hyperparameter_finetuning.md):
  1. Train baseline hyperparameters with early stopping, once per seed in [0, 42, 101]
  2. Random-search trials: sample hparams, train with early stopping per seed
  3. Eval each checkpoint: 50 episodes × seeds [0, 42, 101]
  4. Log JSONL results; plot with kitchen_plot_results.py

Usage:
  conda activate flowpolicy-kitchen
  cd FlowPolicy
  python scripts/kitchen_hparam_search.py --phase all --n-trials 10 --gpu 0
  python scripts/kitchen_hparam_search.py --phase baseline --gpu 0
  python scripts/kitchen_hparam_search.py --phase search --n-trials 5 --gpu 0
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))
os.chdir(PKG_ROOT)

from flow_policy_3d.training.hparam_search import (  # noqa: E402
    EXPERIMENT_SEEDS,
    KitchenHparamSearchSpace,
    aggregate_multiseed_metrics,
    get_baseline_hparams,
    hparams_to_hydra_overrides,
    read_training_summary,
    save_best_config,
    save_trial_result,
)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    trial_id: int,
    hparams: dict,
    gpu: int,
    wandb_mode: str,
    logging_group: str,
) -> Path:
    phase = hparams.get("phase", "search")
    train_seed = hparams["train_seed"]
    if phase == "baseline":
        exp_name = f"kitchen_baseline_seed{train_seed}"
    else:
        exp_name = f"kitchen_hparam_t{trial_id:04d}_seed{train_seed}"
    run_dir = PKG_ROOT / "data" / "outputs" / exp_name
    overrides = hparams_to_hydra_overrides(hparams, str(run_dir))
    overrides.extend(
        [
            f"exp_name={exp_name}",
            f"logging.name={exp_name}",
            f"logging.mode={wandb_mode}",
            f"logging.group={logging_group}",
        ]
    )
    cmd = [
        sys.executable,
        "train.py",
        "--config-name=flowpolicy_kitchen.yaml",
        *overrides,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HYDRA_FULL_ERROR"] = "1"
    print(f"[Trial {trial_id} seed={train_seed}] Training:", " ".join(cmd))
    subprocess.run(cmd, cwd=PKG_ROOT, env=env, check=True)
    return run_dir


def find_best_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    sr_ckpt = ckpt_dir / "best_success_rate.ckpt"
    if sr_ckpt.is_file():
        return sr_ckpt
    fixed = ckpt_dir / "best_val_loss.ckpt"
    if fixed.is_file():
        return fixed
    best = list(ckpt_dir.glob("best_val_loss_epoch*.ckpt"))
    if best:
        return max(best, key=lambda p: p.stat().st_mtime)
    latest = ckpt_dir / "latest.ckpt"
    if latest.is_file():
        return latest
    topk = list(ckpt_dir.glob("epoch=*.ckpt"))
    if topk:
        return max(topk, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")


def eval_one_seed(
    ckpt_path: Path,
    eval_seed: int,
    hparams: dict,
    gpu: int,
    eval_episodes: int = 50,
) -> dict:
    import dill
    from flow_policy_3d.env_runner.kitchen_runner import KitchenRunner
    from train import TrainFlowPolicyWorkspace

    set_global_seed(eval_seed)
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cfg.training.seed = eval_seed
    cfg.task.env_runner.n_obs_steps = hparams["n_obs_steps"]
    cfg.task.env_runner.n_action_steps = hparams["n_action_steps"]
    cfg.task.env_runner.eval_episodes = eval_episodes

    run_dir = str(ckpt_path.parent.parent)
    ws = TrainFlowPolicyWorkspace(cfg, output_dir=run_dir)
    pickle_keys = [
        k for k in payload.get("pickles", {}) if k != "_output_dir"
    ]
    ws.load_payload(payload, include_keys=pickle_keys)
    ws._output_dir = run_dir
    policy = ws.ema_model if cfg.training.use_ema and ws.ema_model else ws.model
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()

    runner = KitchenRunner(
        output_dir=run_dir,
        eval_episodes=eval_episodes,
        n_obs_steps=hparams["n_obs_steps"],
        n_action_steps=hparams["n_action_steps"],
        device=str(device),
    )
    log = runner.run(policy, warmup_gpu=True)
    log["eval_seed"] = eval_seed
    return log


def run_trial(
    trial_id: int,
    hparams: dict | None,
    gpu: int,
    train_seeds: list[int],
    eval_seeds: list[int],
    eval_episodes: int,
    skip_train: bool,
    run_dir: Path | None,
    ckpt_path: Path | None,
    wandb_mode: str,
    results_path: Path,
    best_json: Path,
    phase: str,
) -> dict:
    space = KitchenHparamSearchSpace()
    all_train_runs: list[dict] = []

    if hparams is None:
        sampled = space.sample(random.Random(trial_id))
        sampled["phase"] = phase
    else:
        sampled = dict(hparams)
        sampled["phase"] = phase

    for train_seed in train_seeds:
        hp = dict(sampled)
        hp["train_seed"] = train_seed

        print(f"[Trial {trial_id} phase={phase}] train_seed={train_seed} hparams:", hp)

        if not skip_train:
            group = "kitchen_baseline" if phase == "baseline" else "kitchen_hparam_search"
            run_dir_ts = run_training(trial_id, hp, gpu, wandb_mode, group)
        else:
            assert run_dir is not None or ckpt_path is not None, (
                "--run-dir or --ckpt-path required with --skip-train"
            )
            run_dir_ts = Path(run_dir) if run_dir else None

        if ckpt_path is not None and skip_train:
            ckpt = Path(ckpt_path)
            if run_dir_ts is None:
                run_dir_ts = ckpt.parent.parent
        else:
            ckpt = find_best_checkpoint(run_dir_ts)

        train_summary = read_training_summary(run_dir_ts)
        per_eval = []
        for es in eval_seeds:
            print(
                f"[Trial {trial_id}] train_seed={train_seed} eval_seed={es} "
                f"episodes={eval_episodes}"
            )
            elog = eval_one_seed(ckpt, es, hp, gpu, eval_episodes)
            per_eval.append(elog)

        eval_agg = aggregate_multiseed_metrics(per_eval)
        run_metrics = {**eval_agg, **train_summary, "train_seed": train_seed}
        all_train_runs.append(run_metrics)

    metrics = aggregate_multiseed_metrics(all_train_runs)
    metrics["trade_off_mean"] = metrics.get("trade_off_mean", 0.0)

    # Representative hparams (without per-seed train_seed) for logging
    log_hparams = dict(sampled)
    log_hparams.pop("train_seed", None)
    log_hparams["train_seeds"] = train_seeds
    log_hparams["eval_seeds"] = eval_seeds
    log_hparams["phase"] = phase

    primary_run = str(run_dir_ts) if run_dir_ts else ""
    save_trial_result(results_path, trial_id, log_hparams, metrics, primary_run)

    if phase == "search" and best_json is not None:
        if best_json.is_file():
            import json

            with open(best_json) as f:
                best = json.load(f)
            prev = best.get("metrics", {}).get("trade_off_mean", -1)
        else:
            prev = -1
        if metrics.get("trade_off_mean", 0) > prev:
            save_best_config(best_json, trial_id, log_hparams, metrics)
            print(
                f"[Trial {trial_id}] New best trade_off={metrics.get('trade_off_mean')}"
            )

    return metrics


def run_baseline_phase(gpu, train_seeds, eval_seeds, eval_episodes, wandb_mode, results_path):
    print("=== Phase 1: Baseline training (early stopping) ===")
    for i, seed in enumerate(train_seeds):
        hp = get_baseline_hparams(seed)
        run_trial(
            trial_id=-(i + 1),
            hparams=hp,
            gpu=gpu,
            train_seeds=[seed],
            eval_seeds=eval_seeds,
            eval_episodes=eval_episodes,
            skip_train=False,
            run_dir=None,
            ckpt_path=None,
            wandb_mode=wandb_mode,
            results_path=results_path,
            best_json=None,
            phase="baseline",
        )


def run_search_phase(
    n_trials,
    start_trial,
    gpu,
    train_seeds,
    eval_seeds,
    eval_episodes,
    wandb_mode,
    results_path,
    best_json,
):
    print("=== Phase 2: Random search (early stopping per trial) ===")
    for tid in range(start_trial, start_trial + n_trials):
        run_trial(
            trial_id=tid,
            hparams=None,
            gpu=gpu,
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            eval_episodes=eval_episodes,
            skip_train=False,
            run_dir=None,
            ckpt_path=None,
            wandb_mode=wandb_mode,
            results_path=results_path,
            best_json=best_json,
            phase="search",
        )


def main():
    parser = argparse.ArgumentParser(description="Kitchen FlowPolicy experiment pipeline")
    parser.add_argument(
        "--phase",
        choices=["baseline", "search", "all"],
        default="all",
        help="baseline first, then search, or both",
    )
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--trial-id", type=int, default=None)
    parser.add_argument("--start-trial", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=EXPERIMENT_SEEDS,
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=EXPERIMENT_SEEDS,
    )
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--hparams-json", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument(
        "--baseline-results",
        type=str,
        default="data/hparam_search/baseline_results.csv",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="data/hparam_search/results.csv",
    )
    parser.add_argument(
        "--best-json",
        type=str,
        default="data/hparam_search/best_config.json",
    )
    parser.add_argument(
        "--plot-after",
        action="store_true",
        help="Run kitchen_plot_results.py after experiment",
    )
    args = parser.parse_args()

    baseline_path = PKG_ROOT / args.baseline_results
    search_path = PKG_ROOT / args.results_csv
    best_json = PKG_ROOT / args.best_json

    if args.trial_id is not None:
        import json

        hp = None
        if args.hparams_json:
            with open(args.hparams_json) as f:
                hp = json.load(f)
        phase = hp.get("phase", "search") if hp else "search"
        path = baseline_path if phase == "baseline" else search_path
        run_trial(
            trial_id=args.trial_id,
            hparams=hp,
            gpu=args.gpu,
            train_seeds=args.train_seeds,
            eval_seeds=args.eval_seeds,
            eval_episodes=args.eval_episodes,
            skip_train=args.skip_train,
            run_dir=Path(args.run_dir) if args.run_dir else None,
            ckpt_path=Path(args.ckpt_path) if args.ckpt_path else None,
            wandb_mode=args.wandb_mode,
            results_path=path,
            best_json=best_json if phase == "search" else None,
            phase=phase,
        )
        if args.plot_after:
            _run_plots()
        return

    if args.phase in ("baseline", "all"):
        run_baseline_phase(
            args.gpu,
            args.train_seeds,
            args.eval_seeds,
            args.eval_episodes,
            args.wandb_mode,
            baseline_path,
        )

    if args.phase in ("search", "all"):
        run_search_phase(
            args.n_trials,
            args.start_trial,
            args.gpu,
            args.train_seeds,
            args.eval_seeds,
            args.eval_episodes,
            args.wandb_mode,
            search_path,
            best_json,
        )

    if args.plot_after:
        _run_plots()


def _run_plots():
    plot_script = PKG_ROOT / "scripts" / "kitchen_plot_results.py"
    if plot_script.is_file():
        subprocess.run([sys.executable, str(plot_script)], cwd=PKG_ROOT, check=True)
    else:
        print("Plot script not found:", plot_script)


if __name__ == "__main__":
    main()
