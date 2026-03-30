from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
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
    feature_tail_shape: Optional[tuple[int, ...]] = None
    topology_signature: Optional[tuple[Any, ...]] = None
    hook_call_count: int = 0
    call_expected_shapes: list[tuple[int, ...]] = field(default_factory=list)
    call_branch_signatures: list[Optional[tuple[Any, ...]]] = field(default_factory=list)
    call_batch_labels: list[Optional[tuple[int, ...]]] = field(default_factory=list)
    call_observed_actual: list[bool] = field(default_factory=list)
    call_used_forecast: list[bool] = field(default_factory=list)
    call_actual_features: list[Optional[torch.Tensor]] = field(default_factory=list)
    call_predicted_features: list[Optional[torch.Tensor]] = field(default_factory=list)


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
        self._history_batch_labels: Optional[tuple[int, ...]] = None
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
        self._history_batch_labels = None
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
        self._history_batch_labels = None
        for step in self._active_steps.values():
            step.call_predicted_features = [None] * len(step.call_predicted_features)
            step.call_used_forecast = [False] * len(step.call_used_forecast)
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

    def _is_tail_actual_step(self, solver_step_id: int) -> bool:
        if self._active_run is None:
            return False
        tail_actual_steps = int(self.cfg.tail_actual_steps)
        if tail_actual_steps <= 0:
            return False
        tail_start = max(0, self._active_run.total_steps - tail_actual_steps)
        return int(solver_step_id) >= tail_start

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
        tail_actual_only = self._is_tail_actual_step(int(solver_step_id))
        if (
            not self.forecast_disabled
            and not tail_actual_only
            and int(solver_step_id) >= self.cfg.warmup_steps
            and self.forecaster.ready(self.min_fit_points)
        ):
            ws_floor = max(1, int(math.floor(self.curr_ws)))
            actual_forward = ((self.num_consecutive_cached_steps + 1) % ws_floor) == 0

        if self.forecast_disabled or tail_actual_only or not self.forecaster.ready(self.min_fit_points):
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
        step = self._require_active_step(run_id, solver_step_id)
        return any(step.call_used_forecast)

    @staticmethod
    def _split_branch_signature(
        branch_signature: Optional[tuple[Any, ...]],
    ) -> tuple[tuple[Any, ...], Optional[tuple[int, ...]]]:
        if branch_signature is None:
            return (), None
        topology_entries: list[Any] = []
        batch_labels: Optional[tuple[int, ...]] = None
        for entry in branch_signature:
            if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == "cond_or_uncond":
                try:
                    batch_labels = tuple(int(v) for v in entry[1])
                except Exception:
                    batch_labels = tuple(entry[1])
            else:
                topology_entries.append(entry)
        return tuple(topology_entries), batch_labels

    def register_model_hook_call(
        self,
        run_id: int,
        solver_step_id: int,
        *,
        expected_shape: tuple[int, ...],
        branch_signature: Optional[tuple[Any, ...]] = None,
    ) -> int:
        step = self._require_active_step(run_id, solver_step_id)
        shape = tuple(expected_shape)
        tail_shape = shape[1:]
        topology_signature, batch_labels = self._split_branch_signature(branch_signature)
        step.hook_call_count += 1
        if step.feature_tail_shape is None:
            step.feature_tail_shape = tail_shape
        elif tail_shape != step.feature_tail_shape:
            self._disable_forecasting("model-hook feature shape changed within one solver step")

        if step.topology_signature is None:
            step.topology_signature = topology_signature
        elif topology_signature != step.topology_signature:
            self._disable_forecasting("model-hook branch signature changed within one solver step")

        if batch_labels is not None and len(batch_labels) != shape[0]:
            batch_labels = None

        step.call_expected_shapes.append(shape)
        step.call_branch_signatures.append(branch_signature)
        step.call_batch_labels.append(batch_labels)
        step.call_observed_actual.append(False)
        step.call_used_forecast.append(False)
        step.call_actual_features.append(None)
        step.call_predicted_features.append(None)
        return len(step.call_expected_shapes) - 1

    def observe_actual_feature(
        self,
        run_id: int,
        solver_step_id: int,
        feature: torch.Tensor,
        *,
        call_id: Optional[int] = None,
    ) -> None:
        step = self._require_active_step(run_id, solver_step_id)
        resolved_call_id = self._resolve_call_id(step, call_id)
        step.call_observed_actual[resolved_call_id] = True
        step.call_used_forecast[resolved_call_id] = False
        step.call_predicted_features[resolved_call_id] = None
        step.call_actual_features[resolved_call_id] = feature.detach()

    @staticmethod
    def _reorder_feature_to_labels(
        feature: torch.Tensor,
        source_labels: tuple[int, ...],
        target_labels: tuple[int, ...],
    ) -> Optional[torch.Tensor]:
        if feature.shape[0] != len(source_labels) or len(source_labels) != len(target_labels):
            return None
        if source_labels == target_labels:
            return feature
        if sorted(source_labels) != sorted(target_labels):
            return None
        source_positions: dict[int, deque[int]] = defaultdict(deque)
        for idx, label in enumerate(source_labels):
            source_positions[int(label)].append(idx)
        order = [source_positions[int(label)].popleft() for label in target_labels]
        return feature[order, ...]

    def predict_feature(
        self,
        run_id: int,
        solver_step_id: int,
        *,
        expected_shape: Optional[tuple[int, ...]] = None,
        call_id: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        step = self._require_active_step(run_id, solver_step_id)
        resolved_call_id = self._resolve_call_id(step, call_id)
        if step.decision["actual_forward"]:
            return None
        if self.forecast_disabled or not self.forecaster.ready(self.min_fit_points):
            return None

        target_shape = tuple(expected_shape) if expected_shape is not None else step.call_expected_shapes[resolved_call_id]
        target_batch_labels = step.call_batch_labels[resolved_call_id]
        history_shape = self.forecaster.feature_shape
        if history_shape is not None:
            history_shape = tuple(history_shape)
            if history_shape[1:] != target_shape[1:]:
                self._disable_forecasting("predicted feature shape did not match the current solver-step input")
                return None
            if history_shape[0] != target_shape[0]:
                return None

        if step.hook_call_count > 1:
            return None

        if self._history_batch_labels is not None and target_batch_labels is None:
            return None

        if step.call_predicted_features[resolved_call_id] is None:
            predicted_feature = self.forecaster.predict(
                time_coord=step.time_coord,
                blend_weight=self.cfg.blend_weight,
            )
            if tuple(predicted_feature.shape) != target_shape:
                self._disable_forecasting("predicted feature shape did not match the current solver-step input")
                return None
            if self._history_batch_labels is not None and target_batch_labels is not None:
                reordered = self._reorder_feature_to_labels(
                    predicted_feature,
                    self._history_batch_labels,
                    target_batch_labels,
                )
                if reordered is None:
                    return None
                predicted_feature = reordered
            step.call_predicted_features[resolved_call_id] = predicted_feature

        step.call_used_forecast[resolved_call_id] = True
        return step.call_predicted_features[resolved_call_id]

    def finalize_solver_step(self, run_id: int, solver_step_id: int, *, used_forecast: bool) -> None:
        step = self._require_active_step(run_id, solver_step_id)
        requested_actual_forward = bool(step.decision["actual_forward"])

        observed_actual = any(step.call_observed_actual)
        used_forecast_any = any(step.call_used_forecast)
        if bool(used_forecast) and not observed_actual:
            used_forecast_any = True
        if step.decision["actual_forward"] and not observed_actual:
            self._disable_forecasting("solver step requested an actual forward but no actual feature was observed")
        if not observed_actual and not used_forecast_any:
            self._disable_forecasting("solver step finished without an actual feature or a forecasted feature")

        if observed_actual and used_forecast_any:
            self._disable_forecasting("solver step mixed forecasted and actual model-hook paths")
            used_forecast_any = False

        if used_forecast_any:
            if step.hook_call_count > 1:
                self._disable_forecasting("forecasted solver step re-entered the model hook")
            self.num_consecutive_cached_steps += 1
            self.stats.forecasted_count += 1
            step.decision["actual_forward"] = False
        else:
            actual_parts = [part for part in step.call_actual_features if part is not None]
            actual_labels = [labels for labels, part in zip(step.call_batch_labels, step.call_actual_features) if part is not None]
            if actual_parts and not self.forecast_disabled:
                labeled_parts = [labels is not None for labels in actual_labels]
                combined_feature: Optional[torch.Tensor] = None
                combined_labels: Optional[tuple[int, ...]] = None
                if any(labeled_parts) and not all(labeled_parts):
                    self._disable_forecasting("model-hook batch layout changed within one solver step")
                elif all(labeled_parts):
                    rows: list[tuple[int, int, torch.Tensor]] = []
                    arrival = 0
                    for labels, part in zip(actual_labels, actual_parts):
                        assert labels is not None
                        if part.shape[0] != len(labels):
                            self._disable_forecasting("model-hook batch layout changed within one solver step")
                            break
                        for row_idx, label in enumerate(labels):
                            rows.append((int(label), arrival, part[row_idx : row_idx + 1]))
                            arrival += 1
                    if rows:
                        rows.sort(key=lambda item: (item[0], item[1]))
                        combined_feature = torch.cat([row for _, _, row in rows], dim=0)
                        combined_labels = tuple(label for label, _, _ in rows)
                else:
                    combined_feature = actual_parts[0] if len(actual_parts) == 1 else torch.cat(actual_parts, dim=0)
                    combined_labels = None

                if combined_feature is not None:
                    try:
                        if self._history_batch_labels is None:
                            self._history_batch_labels = combined_labels
                        elif combined_labels is None:
                            self._disable_forecasting("combined actual feature batch layout changed across solver steps")
                        else:
                            reordered = self._reorder_feature_to_labels(
                                combined_feature,
                                combined_labels,
                                self._history_batch_labels,
                            )
                            if reordered is None:
                                self._disable_forecasting("combined actual feature batch layout changed across solver steps")
                            else:
                                combined_feature = reordered
                                combined_labels = self._history_batch_labels

                        if not self.forecast_disabled:
                            self.forecaster.update(step.time_coord, combined_feature)
                    except ValueError:
                        self._disable_forecasting("combined actual feature shape changed across solver steps")
            if (
                requested_actual_forward
                and not self.forecast_disabled
                and step.solver_step_id >= self.cfg.warmup_steps
            ):
                self.curr_ws = round(self.curr_ws + float(self.cfg.flex_window), 6)
            self.num_consecutive_cached_steps = 0
            self.stats.actual_forward_count += 1
            step.decision["actual_forward"] = True

        self.stats.current_window = self.curr_ws
        self._active_steps.pop(int(solver_step_id), None)

    @staticmethod
    def _resolve_call_id(step: _ActiveStep, call_id: Optional[int]) -> int:
        if not step.call_expected_shapes:
            raise RuntimeError("Spectrum solver step has no active model-hook call.")
        resolved = len(step.call_expected_shapes) - 1 if call_id is None else int(call_id)
        if resolved < 0 or resolved >= len(step.call_expected_shapes):
            raise RuntimeError(f"Spectrum solver-step call id {resolved} is not active.")
        return resolved

    def _require_active_step(self, run_id: int, solver_step_id: int) -> _ActiveStep:
        if self._active_run is None or self._active_run.run_id != int(run_id):
            raise RuntimeError("Spectrum runtime is not inside the requested sampling run.")
        step = self._active_steps.get(int(solver_step_id))
        if step is None:
            raise RuntimeError(f"Spectrum solver step {solver_step_id} is not active.")
        return step
