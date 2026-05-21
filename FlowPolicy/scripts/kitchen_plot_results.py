#!/usr/bin/env python3
"""
Plot Franka Kitchen experiment results (baseline + random search).

Reads:
  - data/hparam_search/baseline_results.jsonl
  - data/hparam_search/results.jsonl
  - data/hparam_search/best_config.json (optional)

Outputs PNGs under data/hparam_search/plots/

Usage:
  conda activate flowpolicy-kitchen
  cd FlowPolicy
  python scripts/kitchen_plot_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PKG_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PKG_ROOT))

from flow_policy_3d.training.hparam_search import load_results_jsonl  # noqa: E402


def _get(row: dict, key: str, default=np.nan) -> float:
    v = row.get(key, row.get(f"{key}_mean", default))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _std(row: dict, key: str) -> float:
    v = row.get(f"{key}_std", 0.0)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_all_rows(baseline_path: Path, search_path: Path) -> list[dict]:
    rows = []
    for path in (baseline_path, search_path):
        for r in load_results_jsonl(path):
            r["_source"] = path.stem
            rows.append(r)
    return rows


def plot_success_vs_latency(rows: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        sr = _get(row, "success_rate")
        lat = _get(row, "mean_inference_latency")
        sr_s = _std(row, "success_rate")
        lat_s = _std(row, "mean_inference_latency")
        phase = row.get("phase", "search")
        tid = row.get("trial_id", "?")
        color = "#2ecc71" if phase == "baseline" else "#3498db"
        marker = "s" if phase == "baseline" else "o"
        ax.errorbar(
            lat,
            sr,
            xerr=lat_s,
            yerr=sr_s,
            fmt=marker,
            color=color,
            alpha=0.85,
            capsize=3,
            label=None,
        )
        ax.annotate(
            f"{phase[:3]}{tid}",
            (lat, sr),
            fontsize=7,
            alpha=0.7,
        )
    ax.set_xlabel("Mean inference latency (s)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success rate vs inference latency")
    # legend proxy
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], marker="s", color="#2ecc71", linestyle="None", label="baseline"),
            Line2D([0], [0], marker="o", color="#3498db", linestyle="None", label="search"),
        ]
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "success_rate_vs_latency.png", dpi=150)
    plt.close(fig)


def plot_tradeoff_vs_loss(rows: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        loss = _get(row, "best_val_loss_ema")
        to = _get(row, "trade_off")
        if np.isnan(loss) or np.isnan(to):
            continue
        phase = row.get("phase", "search")
        color = "#2ecc71" if phase == "baseline" else "#e74c3c"
        ax.scatter(loss, to, c=color, s=60, alpha=0.8)
    ax.set_xlabel("Best validation loss (EMA)")
    ax.set_ylabel("Trade-off (success_rate / latency)")
    ax.set_title("Trade-off vs validation loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_vs_val_loss.png", dpi=150)
    plt.close(fig)


def plot_subtask_success(rows: list[dict], out_dir: Path):
    keys = ["success_rate_k1", "success_rate_k2", "success_rate_k3", "success_rate_k4"]
    labels = ["k=1 microwave", "k=2 +light", "k=3 +kettle", "k=4 all tasks"]
    x = np.arange(len(keys))
    width = 0.35
    baseline = [r for r in rows if r.get("phase") == "baseline"]
    search = [r for r in rows if r.get("phase") != "baseline"]
    fig, ax = plt.subplots(figsize=(10, 5))

    def bar_group(group, offset, name, color):
        if not group:
            return
        means = [_get(group[0], k) for k in keys]
        stds = [_std(group[0], k) for k in keys]
        ax.bar(x + offset, means, width, yerr=stds, label=name, color=color, capsize=3)

    if baseline:
        bar_group(baseline, -width / 2, "baseline (mean)", "#2ecc71")
    if search:
        means = np.nanmean([[_get(r, k) for k in keys] for r in search], axis=0)
        ax.bar(x + width / 2, means, width, label="search trials (avg)", color="#3498db")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Per-subtask success rates")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "subtask_success_rates.png", dpi=150)
    plt.close(fig)


def plot_hparam_scatter(rows: list[dict], out_dir: Path):
    """Trade-off vs num_segments and hidden_dim for search trials."""
    search = [r for r in rows if r.get("phase") == "search"]
    if not search:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for row in search:
        to = _get(row, "trade_off")
        axes[0].scatter(row.get("num_segments", np.nan), to, alpha=0.7)
        axes[1].scatter(row.get("hidden_dim", np.nan), to, alpha=0.7)
    axes[0].set_xlabel("num_segments (K)")
    axes[0].set_ylabel("Trade-off")
    axes[0].set_title("Trade-off vs K")
    axes[1].set_xlabel("hidden_dim")
    axes[1].set_ylabel("Trade-off")
    axes[1].set_title("Trade-off vs hidden_dim")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_vs_hparams.png", dpi=150)
    plt.close(fig)


def plot_latency_vs_segments(rows: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        k = row.get("num_segments", np.nan)
        lat = _get(row, "mean_inference_latency")
        sr = _get(row, "success_rate")
        phase = row.get("phase", "search")
        c = "#2ecc71" if phase == "baseline" else "#9b59b6"
        ax.scatter(k, lat, c=c, s=50 + sr, alpha=0.6)
    ax.set_xlabel("num_segments (K)")
    ax.set_ylabel("Mean inference latency (s)")
    ax.set_title("Latency vs K (marker size ∝ success rate)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_num_segments.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-results",
        default="data/hparam_search/baseline_results.csv",
    )
    parser.add_argument("--search-results", default="data/hparam_search/results.csv")
    parser.add_argument("--out-dir", default="data/hparam_search/plots")
    args = parser.parse_args()

    baseline_path = PKG_ROOT / args.baseline_results
    search_path = PKG_ROOT / args.search_results
    out_dir = PKG_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_all_rows(baseline_path, search_path)
    if not rows:
        print("No results found. Run the experiment first.")
        print("  baseline:", baseline_path.with_suffix(".jsonl"))
        print("  search:  ", search_path.with_suffix(".jsonl"))
        return

    print(f"Loaded {len(rows)} result rows -> {out_dir}")
    plot_success_vs_latency(rows, out_dir)
    plot_tradeoff_vs_loss(rows, out_dir)
    plot_subtask_success(rows, out_dir)
    plot_hparam_scatter(rows, out_dir)
    plot_latency_vs_segments(rows, out_dir)

    summary = {
        "n_rows": len(rows),
        "plots": [
            "success_rate_vs_latency.png",
            "tradeoff_vs_val_loss.png",
            "subtask_success_rates.png",
            "tradeoff_vs_hparams.png",
            "latency_vs_num_segments.png",
        ],
    }
    with open(out_dir / "plot_manifest.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved plots:", ", ".join(summary["plots"]))


if __name__ == "__main__":
    main()
