from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .config import SpectrumConfig
from .forecast import ChebyshevSpectrumForecaster


@dataclass(slots=True)
class RuntimeStats:
    actual_forward_count: int = 0
    forecasted_count: int = 0
    total_steps: int = 0
    current_window: float = 0.0
    last_sigma: Optional[float] = None
    run_id: int = 0


class SpectrumRuntime:
    def __init__(self, cfg: SpectrumConfig):
        self.cfg = cfg.validate()
        self.forecaster = ChebyshevSpectrumForecaster(
            degree=self.cfg.degree,
            ridge_lambda=self.cfg.ridge_lambda,
            max_history=self.cfg.max_history,
        )
        self._last_schedule_signature: Optional[tuple] = None
        self.run_id = 0
        self.stats = RuntimeStats(current_window=float(self.cfg.window_size))
        self.reset_cycle(reset_schedule=False)

    @property
    def min_fit_points(self) -> int:
        return max(2, self.cfg.degree + 1)

    def reset_cycle(self, reset_schedule: bool = False) -> None:
        self.step_idx = 0
        self.curr_ws = float(self.cfg.window_size)
        self.num_consecutive_cached_steps = 0
        self.decisions_by_sigma: Dict[float, Dict[str, Any]] = {}
        self.seen_sigmas: List[float] = []
        self.cycle_finished = False
        self.forecaster.reset()
        if reset_schedule:
            self._last_schedule_signature = None
        self.stats.current_window = float(self.cfg.window_size)

    def reset_all(self) -> None:
        self.run_id += 1
        self.stats = RuntimeStats(current_window=float(self.cfg.window_size), run_id=self.run_id)
        self.reset_cycle(reset_schedule=True)

    def _schedule_signature(self, transformer_options: Dict[str, Any]) -> Optional[tuple]:
        sample_sigmas = transformer_options.get("sample_sigmas")
        if sample_sigmas is None:
            return None
        try:
            values = sample_sigmas.detach().float().cpu().flatten().tolist()
            return tuple(round(float(v), 8) for v in values)
        except Exception:
            return None

    def _ensure_run_sync(self, transformer_options: Dict[str, Any]) -> None:
        signature = self._schedule_signature(transformer_options)
        if signature is None:
            return
        if self._last_schedule_signature is None:
            self._last_schedule_signature = signature
            self.stats.total_steps = max(len(signature) - 1, 1)
            return
        if signature != self._last_schedule_signature:
            self.run_id += 1
            self._last_schedule_signature = signature
            self.stats.actual_forward_count = 0
            self.stats.forecasted_count = 0
            self.stats.total_steps = max(len(signature) - 1, 1)
            self.stats.run_id = self.run_id
            self.reset_cycle(reset_schedule=False)

    def num_steps(self) -> int:
        if self.stats.total_steps > 0:
            return self.stats.total_steps
        return 50

    def _sigma_key(self, transformer_options: Dict[str, Any], timesteps: torch.Tensor) -> float:
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None:
            try:
                return round(float(sigmas.detach().flatten()[0].item()), 8)
            except Exception:
                pass
        try:
            return round(float(timesteps.detach().flatten()[0].item()), 8)
        except Exception:
            return float(self.step_idx)

    def _finish_cycle_if_needed(self) -> None:
        if len(self.seen_sigmas) >= self.num_steps() and not self.cycle_finished:
            self.cycle_finished = True

    def _restart_cycle(self) -> None:
        self.run_id += 1
        self.stats.actual_forward_count = 0
        self.stats.forecasted_count = 0
        self.stats.run_id = self.run_id
        self.reset_cycle(reset_schedule=False)

    def _should_restart_on_sigma(self, sigma: float) -> bool:
        if not self.seen_sigmas:
            return False
        if sigma != self.seen_sigmas[0]:
            return False
        return len(self.seen_sigmas) > 1

    def begin_step(self, transformer_options: Dict[str, Any], timesteps: torch.Tensor) -> Dict[str, Any]:
        transformer_options = transformer_options or {}
        self._ensure_run_sync(transformer_options)

        sigma = self._sigma_key(transformer_options, timesteps)
        self.stats.last_sigma = sigma
        self._finish_cycle_if_needed()

        if self.cycle_finished or self._should_restart_on_sigma(sigma):
            self._restart_cycle()

        if sigma in self.decisions_by_sigma:
            return self.decisions_by_sigma[sigma]

        step_idx = len(self.seen_sigmas)
        self.seen_sigmas.append(sigma)

        actual_forward = True
        if step_idx >= self.cfg.warmup_steps and self.forecaster.ready(self.min_fit_points):
            ws_floor = max(1, int(math.floor(self.curr_ws)))
            actual_forward = ((self.num_consecutive_cached_steps + 1) % ws_floor) == 0

        if not self.forecaster.ready(self.min_fit_points):
            actual_forward = True

        if actual_forward:
            self.num_consecutive_cached_steps = 0
            if step_idx >= self.cfg.warmup_steps:
                self.curr_ws = round(self.curr_ws + float(self.cfg.flex_window), 6)
            self.stats.actual_forward_count += 1
        else:
            self.num_consecutive_cached_steps += 1
            self.stats.forecasted_count += 1

        self.step_idx = step_idx
        self.stats.current_window = self.curr_ws

        decision = {
            "sigma": sigma,
            "step_idx": step_idx,
            "total_steps": self.num_steps(),
            "actual_forward": actual_forward,
            "run_id": self.run_id,
        }
        self.decisions_by_sigma[sigma] = decision
        return decision
