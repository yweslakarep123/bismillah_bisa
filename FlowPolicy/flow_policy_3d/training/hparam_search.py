"""Hyperparameter search utilities for Franka Kitchen (see flowpolicy_hyperparameter_finetuning.md)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Seeds for the full experiment (training + eval); report mean ± std across these.
EXPERIMENT_SEEDS: List[int] = [0, 42, 101]


def compute_trajectory_horizon(n_obs_steps: int, n_action_steps: int) -> int:
    """horizon = 4 * ((max(n_obs + n_action - 1, 4) + 3) // 4)"""
    inner = max(n_obs_steps + n_action_steps - 1, 4)
    return 4 * ((inner + 3) // 4)


@dataclass
class KitchenHparamSearchSpace:
    """Exploration ranges from flowpolicy_hyperparameter_finetuning.md §2."""

    num_epochs: List[int] = field(
        default_factory=lambda: [500, 1000, 3000, 5000]
    )
    learning_rate: List[float] = field(
        default_factory=lambda: [1e-3, 5e-4, 1e-4, 1e-5]
    )
    batch_size: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    hidden_dim: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    time_embedding_dim: List[int] = field(
        default_factory=lambda: [128, 256, 512, 1024]
    )
    num_segments: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    epsilon: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1.0])
    delta_t: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1.0])
    n_action_steps: List[int] = field(default_factory=lambda: [2, 4, 6, 8])
    n_obs_steps: List[int] = field(default_factory=lambda: [4, 6, 8, 16])
    def sample(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        rng = rng or random.Random()
        n_obs = rng.choice(self.n_obs_steps)
        n_action = rng.choice(self.n_action_steps)
        hidden = rng.choice(self.hidden_dim)
        return {
            "num_epochs": rng.choice(self.num_epochs),
            "learning_rate": rng.choice(self.learning_rate),
            "batch_size": rng.choice(self.batch_size),
            "hidden_dim": hidden,
            "time_embedding_dim": rng.choice(self.time_embedding_dim),
            "num_segments": rng.choice(self.num_segments),
            "epsilon": rng.choice(self.epsilon),
            "delta_t": rng.choice(self.delta_t),
            "n_obs_steps": n_obs,
            "n_action_steps": n_action,
            "horizon": compute_trajectory_horizon(n_obs, n_action),
        }


def get_baseline_hparams(train_seed: int) -> Dict[str, Any]:
    """Default baseline hyperparameters (§1 doc); train_seed set per run."""
    return {
        "num_epochs": 3000,
        "learning_rate": 1e-4,
        "batch_size": 128,
        "hidden_dim": 512,
        "time_embedding_dim": 256,
        "num_segments": 2,
        "epsilon": 1e-2,
        "delta_t": 1e-2,
        "n_obs_steps": 2,
        "n_action_steps": 4,
        "horizon": 8,
        "train_seed": train_seed,
        "phase": "baseline",
    }


def hparams_to_hydra_overrides(hparams: Dict[str, Any], run_dir: str) -> List[str]:
    """Convert sampled dict to Hydra CLI overrides for flowpolicy_kitchen.yaml."""
    hd = hparams["hidden_dim"]
    overrides = [
        f"hydra.run.dir={run_dir}",
        f"training.num_epochs={hparams['num_epochs']}",
        f"training.seed={hparams['train_seed']}",
        f"optimizer.lr={hparams['learning_rate']}",
        f"dataloader.batch_size={hparams['batch_size']}",
        f"val_dataloader.batch_size={hparams['batch_size']}",
        f"horizon={hparams['horizon']}",
        f"n_obs_steps={hparams['n_obs_steps']}",
        f"n_action_steps={hparams['n_action_steps']}",
        f"policy.encoder_output_dim={hd}",
        f"policy.state_mlp_size=[{hd},{hd}]",
        f"policy.diffusion_step_embed_dim={hparams['time_embedding_dim']}",
        f"policy.Conditional_ConsistencyFM.num_segments={hparams['num_segments']}",
        f"policy.Conditional_ConsistencyFM.eps={hparams['epsilon']}",
        f"policy.Conditional_ConsistencyFM.delta={hparams['delta_t']}",
        f"task.dataset.pad_before={hparams['n_obs_steps'] - 1}",
        f"task.dataset.pad_after={hparams['n_action_steps'] - 1}",
        f"task.env_runner.n_obs_steps={hparams['n_obs_steps']}",
        f"task.env_runner.n_action_steps={hparams['n_action_steps']}",
        "checkpoint.save_ckpt=true",
        "training.use_early_stopping=true",
        "training.run_validation=true",
    ]
    return overrides


_METRIC_KEYS = [
    "success_rate",
    "mean_inference_latency",
    "trade_off",
    "success_rate_k1",
    "success_rate_k2",
    "success_rate_k3",
    "success_rate_k4",
    "best_val_loss_ema",
    "stopped_epoch",
]


def aggregate_multiseed_metrics(
    per_seed_logs: List[Dict[str, float]],
) -> Dict[str, float]:
    """Mean ± std across eval (or train) seeds for key metrics."""
    import numpy as np

    out: Dict[str, float] = {}
    for key in _METRIC_KEYS:
        vals = []
        for log in per_seed_logs:
            if key in log:
                vals.append(log[key])
            elif f"{key}_mean" in log:
                vals.append(log[f"{key}_mean"])
        if not vals:
            continue
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def load_results_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load experiment results (results.jsonl or baseline_results.jsonl)."""
    jsonl = path.with_suffix(".jsonl") if path.suffix == ".csv" else path
    if not jsonl.is_file():
        return []
    rows = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_training_summary(run_dir: Path) -> Dict[str, float]:
    """Read val-loss / epoch summary written by train.py."""
    summary_path = run_dir / "training_summary.json"
    if not summary_path.is_file():
        return {}
    with open(summary_path) as f:
        data = json.load(f)
    return {
        k: float(v)
        for k, v in data.items()
        if isinstance(v, (int, float))
    }


def save_trial_result(
    results_path: Path,
    trial_id: int,
    hparams: Dict[str, Any],
    metrics: Dict[str, float],
    run_dir: str,
):
    """Append one trial as JSONL (robust when columns vary between trials)."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "trial_id": trial_id,
        "run_dir": run_dir,
        **{k: v for k, v in hparams.items() if k != "phase"},
        "phase": hparams.get("phase", "search"),
        **metrics,
    }
    jsonl_path = results_path
    if results_path.suffix == ".csv":
        jsonl_path = results_path.with_suffix(".jsonl")
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def save_best_config(path: Path, trial_id: int, hparams: Dict, metrics: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {"trial_id": trial_id, "hparams": hparams, "metrics": metrics},
            f,
            indent=2,
        )
