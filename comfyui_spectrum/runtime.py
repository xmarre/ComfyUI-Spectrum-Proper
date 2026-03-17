from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .config import SpectrumConfig
from .forecast import ChebyshevSpectrumForecaster


@dataclass(slots=True)
class RuntimeStats:
    actual_forward_count: int = 0
    forecasted_count: int = 0
    total_steps: int = 0
    current_window: float = 0.0
    run_id: int = 0
    forecast_disabled: bool = False
    disable_reason: Optional[str] = None
    sampler_name: Optional[str] = None


@dataclass(slots=True)
class _ActiveRun:
    run_id: int
    sampler_name: str
    total_steps: int
    schedule_values: tuple[float, ...]
    schedule_coords: tuple[float, ...]
    supports_solver_steps: bool
    next_solver_step_id: int = 0


@dataclass(slots=True)
class _ActiveStep:
    solver_step_id: int
    time_coord: float
    decision: Dict[str, Any]
    expected_shape: Optional[tuple[int, ...]] = None
    branch_signature: Optional[tuple[Any, ...]] = None
    hook_call_count: int = 0
    observed_actual: bool = False
    used_forecast: bool = False
    predicted_feature: Optional[torch.Tensor] = None


class SpectrumRuntime:
    def __init__(self, cfg: SpectrumConfig):
        self.cfg = cfg.validate()
        self.forecaster = ChebyshevSpectrumForecaster(
            degree=self.cfg.degree,
            ridge_lambda=self.cfg.ridge_lambda,
            max_history=self.cfg.max_history,
        )
        self.run_id = 0
        self.stats = RuntimeStats(current_window=float(self.cfg.window_size))
        self._active_run: Optional[_ActiveRun] = None
        self._active_steps: Dict[int, _ActiveStep] = {}
        self._reset_scheduler_state()

    @property
    def min_fit_points(self) -> int:
        return max(2, self.cfg.degree + 1)

    @property
    def active_run_id(self) -> Optional[int]:
        if self._active_run is None:
            return None
        return self._active_run.run_id

    @property
    def next_solver_step_id(self) -> int:
        if self._active_run is None:
            return 0
        return self._active_run.next_solver_step_id

    @property
    def active_run_supports_solver_steps(self) -> bool:
        if self._active_run is None:
            return False
        return bool(self._active_run.supports_solver_steps)

    def _reset_scheduler_state(self) -> None:
        self.curr_ws = float(self.cfg.window_size)
        self.num_consecutive_cached_steps = 0
        self.forecast_disabled = False
        self.forecast_disable_reason: Optional[str] = None
        self.forecaster.reset()
        self._active_steps = {}
        self.stats.current_window = float(self.cfg.window_size)
        self.stats.forecast_disabled = False
        self.stats.disable_reason = None

    def reset_all(self) -> None:
        self.run_id += 1
        self.stats = RuntimeStats(current_window=float(self.cfg.window_size), run_id=self.run_id)
        self._active_run = None
        self._reset_scheduler_state()

    def _disable_forecasting(self, reason: str) -> None:
        if self.forecast_disabled:
            return
        self.forecast_disabled = True
        self.forecast_disable_reason = reason
        self.num_consecutive_cached_steps = 0
        self.curr_ws = float(self.cfg.window_size)
        self.forecaster.reset()
        for step in self._active_steps.values():
            step.predicted_feature = None
        self.stats.current_window = self.curr_ws
        self.stats.forecast_disabled = True
        self.stats.disable_reason = reason

    def num_steps(self) -> int:
        if self.stats.total_steps > 0:
            return self.stats.total_steps
        return 50

    @staticmethod
    def _build_schedule_coords(sample_sigmas: torch.Tensor) -> tuple[tuple[float, ...], tuple[float, ...]]:
        values = tuple(float(v) for v in sample_sigmas.detach().flatten().tolist()[:-1])
        if not values:
            return (), ()

        start = values[0]
        end = values[-1]
        denom = end - start
        if abs(denom) < 1e-12:
            coords = tuple(0.0 for _ in values)
        else:
            coords = tuple(((v - start) / denom) * 2.0 - 1.0 for v in values)
        return values, coords

    def time_coord_for_step(self, solver_step_id: int) -> float:
        if self._active_run is None:
            raise RuntimeError("Spectrum runtime is not inside an active run.")
        idx = int(solver_step_id)
        if idx < 0 or idx >= len(self._active_run.schedule_coords):
            raise RuntimeError(f"Spectrum solver step {solver_step_id} is outside the active schedule.")
        return float(self._active_run.schedule_coords[idx])

    def start_run(self, sample_sigmas: torch.Tensor, sampler_name: str, *, supports_solver_steps: bool) -> int:
        self.run_id += 1
        schedule_values, schedule_coords = self._build_schedule_coords(sample_sigmas)
        total_steps = max(len(schedule_coords), 1)
        self.stats = RuntimeStats(
            current_window=float(self.cfg.window_size),
            total_steps=total_steps,
            run_id=self.run_id,
            sampler_name=sampler_name,
        )
        self._active_run = _ActiveRun(
            run_id=self.run_id,
            sampler_name=sampler_name,
            total_steps=total_steps,
            schedule_values=schedule_values,
            schedule_coords=schedule_coords,
            supports_solver_steps=bool(supports_solver_steps),
        )
        self._reset_scheduler_state()
        self.stats.run_id = self.run_id
        self.stats.total_steps = total_steps
        self.stats.sampler_name = sampler_name
        if not supports_solver_steps:
            self._disable_forecasting(
                f"sampler {sampler_name!r} does not expose one predict_noise call per solver step"
            )
        return self.run_id

    def end_run(self, run_id: int) -> None:
        if self._active_run is None or self._active_run.run_id != int(run_id):
            return
        self._active_run = None
        self._active_steps = {}

    def begin_solver_step(self, run_id: int, solver_step_id: int, time_coord: float, total_steps: int) -> Dict[str, Any]:
        if self._active_run is None or self._active_run.run_id != int(run_id):
            raise RuntimeError("Spectrum runtime is not inside the requested sampling run.")

        if int(total_steps) != self._active_run.total_steps:
            self._disable_forecasting("solver-step total_steps changed inside one sampling run")

        expected_time_coord = self.time_coord_for_step(int(solver_step_id))
        if not math.isclose(float(time_coord), expected_time_coord, rel_tol=0.0, abs_tol=1e-8):
            self._disable_forecasting("solver-step time_coord did not match the active schedule")

        existing = self._active_steps.get(int(solver_step_id))
        if existing is not None:
            return existing.decision

        if int(solver_step_id) != self._active_run.next_solver_step_id:
            self._disable_forecasting("solver-step ids are not sequential within the sampling run")

        actual_forward = True
        if (
            not self.forecast_disabled
            and int(solver_step_id) >= self.cfg.warmup_steps
            and self.forecaster.ready(self.min_fit_points)
        ):
            ws_floor = max(1, int(math.floor(self.curr_ws)))
            actual_forward = ((self.num_consecutive_cached_steps + 1) % ws_floor) == 0

        if self.forecast_disabled or not self.forecaster.ready(self.min_fit_points):
            actual_forward = True

        decision = {
            "run_id": int(run_id),
            "solver_step_id": int(solver_step_id),
            "time_coord": float(time_coord),
            "total_steps": int(total_steps),
            "actual_forward": bool(actual_forward),
            "forecast_disabled": self.forecast_disabled,
        }
        self._active_steps[int(solver_step_id)] = _ActiveStep(
            solver_step_id=int(solver_step_id),
            time_coord=float(time_coord),
            decision=decision,
        )
        self._active_run.next_solver_step_id = int(solver_step_id) + 1
        self.stats.current_window = self.curr_ws
        return decision

    def get_step_decision(self, run_id: int, solver_step_id: int) -> Optional[Dict[str, Any]]:
        step = self._active_steps.get(int(solver_step_id))
        if step is None or step.decision["run_id"] != int(run_id):
            return None
        return step.decision

    def step_used_forecast(self, run_id: int, solver_step_id: int) -> bool:
        return self._require_active_step(run_id, solver_step_id).used_forecast

    def register_model_hook_call(
        self,
        run_id: int,
        solver_step_id: int,
        *,
        expected_shape: tuple[int, ...],
        branch_signature: Optional[tuple[Any, ...]] = None,
    ) -> None:
        step = self._require_active_step(run_id, solver_step_id)
        step.hook_call_count += 1
        if step.hook_call_count > 1:
            self._disable_forecasting("multiple model-hook calls observed within one solver step")
        if step.expected_shape is None:
            step.expected_shape = tuple(expected_shape)
        elif tuple(expected_shape) != step.expected_shape:
            self._disable_forecasting("model-hook feature shape changed within one solver step")

        if branch_signature is None:
            return
        if step.branch_signature is None:
            step.branch_signature = branch_signature
        elif branch_signature != step.branch_signature:
            self._disable_forecasting("model-hook branch signature changed within one solver step")

    def observe_actual_feature(self, run_id: int, solver_step_id: int, feature: torch.Tensor) -> None:
        step = self._require_active_step(run_id, solver_step_id)
        step.observed_actual = True
        step.used_forecast = False
        step.predicted_feature = None
        if self.forecast_disabled:
            return
        self.forecaster.update(step.time_coord, feature)

    def predict_feature(
        self,
        run_id: int,
        solver_step_id: int,
        *,
        expected_shape: Optional[tuple[int, ...]] = None,
    ) -> Optional[torch.Tensor]:
        step = self._require_active_step(run_id, solver_step_id)
        if self.forecast_disabled or not self.forecaster.ready(self.min_fit_points):
            return None

        if step.predicted_feature is None:
            step.predicted_feature = self.forecaster.predict(
                time_coord=step.time_coord,
                blend_weight=self.cfg.blend_weight,
            )

        if expected_shape is not None and tuple(step.predicted_feature.shape) != tuple(expected_shape):
            self._disable_forecasting("predicted feature shape did not match the current solver-step input")
            return None

        step.used_forecast = True
        return step.predicted_feature

    def finalize_solver_step(self, run_id: int, solver_step_id: int, *, used_forecast: bool) -> None:
        step = self._require_active_step(run_id, solver_step_id)
        if bool(used_forecast):
            step.used_forecast = True
        if step.decision["actual_forward"] and not step.observed_actual:
            self._disable_forecasting("solver step requested an actual forward but no actual feature was observed")
        if not step.observed_actual and not step.used_forecast:
            self._disable_forecasting("solver step finished without an actual feature or a forecasted feature")

        if step.used_forecast:
            self.num_consecutive_cached_steps += 1
            self.stats.forecasted_count += 1
            step.decision["actual_forward"] = False
        else:
            if not self.forecast_disabled and step.solver_step_id >= self.cfg.warmup_steps:
                self.curr_ws = round(self.curr_ws + float(self.cfg.flex_window), 6)
            self.num_consecutive_cached_steps = 0
            self.stats.actual_forward_count += 1
            step.decision["actual_forward"] = True

        self.stats.current_window = self.curr_ws
        self._active_steps.pop(int(solver_step_id), None)

    def _require_active_step(self, run_id: int, solver_step_id: int) -> _ActiveStep:
        if self._active_run is None or self._active_run.run_id != int(run_id):
            raise RuntimeError("Spectrum runtime is not inside the requested sampling run.")
        step = self._active_steps.get(int(solver_step_id))
        if step is None:
            raise RuntimeError(f"Spectrum solver step {solver_step_id} is not active.")
        return step
