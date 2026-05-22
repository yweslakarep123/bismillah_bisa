"""Conservative early stopping for small Franka Kitchen dataset."""

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Union


def _extract_success_metrics(
    metrics: Union[float, Mapping[str, float]],
) -> Dict[str, float]:
    """Normalize rollout metrics to tracked success-rate keys."""
    if isinstance(metrics, (int, float)):
        return {"success_rate": float(metrics)}

    tracked: Dict[str, float] = {}
    for key, value in metrics.items():
        if key in ("success_rate", "test_mean_score", "mean_success_rates"):
            tracked["success_rate"] = float(value)
        elif key.startswith("success_rate_k"):
            tracked[key] = float(value)
    if "success_rate" not in tracked and "test_mean_score" in metrics:
        tracked["success_rate"] = float(metrics["test_mean_score"])
    return tracked


@dataclass
class EarlyStoppingState:
    val_loss_ema: Optional[float] = None
    best_val_loss_ema: float = float("inf")
    val_loss_patience_counter: int = 0
    best_success_rate: float = 0.0
    best_success_rates: Dict[str, float] = field(default_factory=dict)
    success_rate_patience_counter: int = 0
    should_stop: bool = False
    signal_val_loss: bool = False
    signal_success_rate: bool = False
    last_improved_success_keys: tuple = ()


@dataclass
class EarlyStoppingConfig:
    min_epochs: int = 1000
    val_loss_patience: int = 400
    val_loss_min_delta: float = 5e-5
    val_loss_ema_alpha: float = 0.05
    success_rate_check_interval: int = 500
    success_rate_patience: int = 2
    success_rate_min_delta: float = 2.0
    success_rate_eval_episodes: int = 20
    track_subtask_success_rates: bool = True


class EarlyStoppingManager:
    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        self.config = config or EarlyStoppingConfig()
        self.state = EarlyStoppingState()

    def update_val_loss(self, epoch: int, val_loss: float) -> bool:
        """Update EMA val loss. Returns True if new best."""
        cfg = self.config
        st = self.state
        alpha = cfg.val_loss_ema_alpha
        if st.val_loss_ema is None:
            st.val_loss_ema = val_loss
        else:
            st.val_loss_ema = alpha * val_loss + (1 - alpha) * st.val_loss_ema

        improved = False
        if st.best_val_loss_ema - st.val_loss_ema >= cfg.val_loss_min_delta:
            st.best_val_loss_ema = st.val_loss_ema
            st.val_loss_patience_counter = 0
            improved = True
        elif epoch >= cfg.min_epochs:
            st.val_loss_patience_counter += 1

        if (
            epoch >= cfg.min_epochs
            and st.val_loss_patience_counter >= cfg.val_loss_patience
        ):
            st.signal_val_loss = True
        return improved

    def update_success_rate(
        self,
        epoch: int,
        metrics: Union[float, Mapping[str, float]],
    ) -> bool:
        """Returns True when any tracked success metric improves."""
        cfg = self.config
        st = self.state
        if epoch < cfg.min_epochs:
            return False

        parsed = _extract_success_metrics(metrics)
        if not parsed:
            return False

        if not cfg.track_subtask_success_rates:
            parsed = {
                k: v for k, v in parsed.items() if k == "success_rate"
            }

        improved_keys = []
        for key, value in parsed.items():
            best = st.best_success_rates.get(key, 0.0)
            if value > best + cfg.success_rate_min_delta:
                st.best_success_rates[key] = value
                improved_keys.append(key)

        if "success_rate" in st.best_success_rates:
            st.best_success_rate = st.best_success_rates["success_rate"]

        if improved_keys:
            st.success_rate_patience_counter = 0
            st.last_improved_success_keys = tuple(improved_keys)
            return True

        st.success_rate_patience_counter += 1
        st.last_improved_success_keys = ()
        if st.success_rate_patience_counter >= cfg.success_rate_patience:
            st.signal_success_rate = True
        return False

    def should_run_success_check(self, epoch: int) -> bool:
        cfg = self.config
        return (
            epoch >= cfg.min_epochs
            and epoch % cfg.success_rate_check_interval == 0
        )

    def check_stop(self, epoch: int) -> bool:
        cfg = self.config
        st = self.state
        if epoch < cfg.min_epochs:
            return False
        if st.signal_val_loss and st.signal_success_rate:
            st.should_stop = True
        return st.should_stop

    def format_success_summary(self) -> str:
        """Human-readable snapshot of tracked success metrics."""
        st = self.state
        if not st.best_success_rates:
            return f"overall={st.best_success_rate:.2f}%"
        parts = []
        for key in sorted(st.best_success_rates.keys()):
            if key == "success_rate":
                parts.append(f"all={st.best_success_rates[key]:.2f}%")
            elif key.startswith("success_rate_k"):
                task_idx = key.replace("success_rate_k", "")
                parts.append(f"k{task_idx}={st.best_success_rates[key]:.2f}%")
        return ", ".join(parts)
