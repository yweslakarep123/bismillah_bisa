"""Conservative early stopping for small Franka Kitchen dataset."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EarlyStoppingState:
    val_loss_ema: Optional[float] = None
    best_val_loss_ema: float = float("inf")
    val_loss_patience_counter: int = 0
    best_success_rate: float = 0.0
    success_rate_patience_counter: int = 0
    should_stop: bool = False
    signal_val_loss: bool = False
    signal_success_rate: bool = False


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

    def update_success_rate(self, epoch: int, success_rate: float) -> bool:
        """Returns True when success rate improves by at least min_delta (pp)."""
        cfg = self.config
        st = self.state
        if epoch < cfg.min_epochs:
            return False
        improved = False
        if success_rate > st.best_success_rate + cfg.success_rate_min_delta:
            st.best_success_rate = success_rate
            st.success_rate_patience_counter = 0
            improved = True
        else:
            st.success_rate_patience_counter += 1
        if st.success_rate_patience_counter >= cfg.success_rate_patience:
            st.signal_success_rate = True
        return improved

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
