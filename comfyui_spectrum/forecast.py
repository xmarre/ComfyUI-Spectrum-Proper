from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(slots=True)
class _HistoryEntry:
    time_coord: float
    feature_flat: torch.Tensor


class ChebyshevSpectrumForecaster:
    """Online Chebyshev forecaster with optional linear blending.

    The forecaster operates on the final hidden feature of the denoiser, not on
    the denoised output itself. This matches the official Spectrum integration
    strategy for FLUX and is the main reason this port stays on the model path
    instead of wrapping the whole sampler output.
    """

    def __init__(self, degree: int = 4, ridge_lambda: float = 0.1, max_history: int = 128):
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self.max_history = int(max_history)
        self.reset()

    def reset(self) -> None:
        self._history: List[_HistoryEntry] = []
        self._feature_shape: Optional[torch.Size] = None
        self._feature_dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None
        self._predict_device: Optional[torch.device] = None
        self._predict_dtype: Optional[torch.dtype] = None
        self._output_device: Optional[torch.device] = None
        self._coeff: Optional[torch.Tensor] = None
        self._coeff_device: Optional[torch.Tensor] = None
        self._cached_degree: Optional[int] = None
        self._cache_dirty = True
        self._gram: Optional[torch.Tensor] = None
        self._rhs: Optional[torch.Tensor] = None
        self._previous_feature_flat_device: Optional[torch.Tensor] = None
        self._previous_time_coord: Optional[float] = None
        self._latest_feature_flat_device: Optional[torch.Tensor] = None
        self._latest_time_coord: Optional[float] = None
        self._linear_mirrors_enabled = True

    def configure(self, degree: int, ridge_lambda: float, max_history: int) -> None:
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self.max_history = int(max_history)
        if self.max_history < 0:
            raise ValueError("max_history must be non-negative.")
        if self.max_history == 0:
            self._history = []
        elif len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]
        self._coeff = None
        self._cached_degree = None
        self._cache_dirty = True
        self._rebuild_stats()
        self._refresh_prediction_mirrors()
        self._recompute_coeff()

    @property
    def feature_shape(self) -> Optional[torch.Size]:
        return self._feature_shape

    def ready(self, min_points: Optional[int] = None) -> bool:
        needed = max(2, int(min_points) if min_points is not None else self.degree + 1)
        return len(self._history) >= needed

    def update(
        self,
        time_coord: float,
        feature: torch.Tensor,
        *,
        predict_device: Optional[torch.device] = None,
        output_device: Optional[torch.device] = None,
        output_dtype: Optional[torch.dtype] = None,
        blend_weight: float = 1.0,
    ) -> None:
        feat = feature.detach()
        if self._feature_shape is None:
            self._feature_shape = feat.shape
        elif feat.shape != self._feature_shape:
            raise ValueError(
                f"Spectrum feature shape changed from {tuple(self._feature_shape)} to {tuple(feat.shape)}."
            )
        resolved_output_dtype = output_dtype if output_dtype is not None else feat.dtype
        resolved_predict_device = predict_device if predict_device is not None else feat.device
        resolved_predict_dtype = self._resolve_predict_dtype(resolved_output_dtype)
        resolved_stats_device = resolved_predict_device if resolved_predict_device is not None else feat.device
        previous_predict_device = self._predict_device
        previous_predict_dtype = self._predict_dtype
        previous_stats_device = self._device
        self._feature_dtype = resolved_output_dtype
        self._predict_device = resolved_predict_device
        if output_device is not None:
            self._output_device = output_device
        else:
            self._output_device = feat.device
        if self._predict_device is None:
            self._predict_device = self._output_device
        if resolved_stats_device is None:
            resolved_stats_device = self._predict_device
        self._device = resolved_stats_device
        self._predict_dtype = resolved_predict_dtype

        linear_mirrors_enabled = float(blend_weight) < (1.0 - 1e-12)
        predict_context_changed = (
            previous_predict_device != self._predict_device or previous_predict_dtype != self._predict_dtype
        )
        if previous_stats_device != self._device:
            self._coeff = None
            self._coeff_device = None
            self._cached_degree = None
            self._cache_dirty = True
            self._rebuild_stats()

        feature_flat = feat.reshape(-1).to(device=self._device, dtype=torch.float32, copy=False)
        entry = _HistoryEntry(float(time_coord), self._archive_feature_for_history(feat))
        self._history.append(entry)
        self._ensure_stats_initialized(feature_flat.numel())
        self._accumulate_feature(time_coord=entry.time_coord, feature_flat=feature_flat, sign=1.0)
        if len(self._history) > self.max_history:
            oldest = self._history.pop(0)
            self._accumulate_entry(oldest, sign=-1.0)
        self._sync_linear_prediction_mirrors(
            entry,
            feat=feat,
            linear_mirrors_enabled=linear_mirrors_enabled,
            force_rebuild=predict_context_changed or linear_mirrors_enabled != self._linear_mirrors_enabled,
        )
        self._coeff = None
        self._coeff_device = None
        self._cached_degree = None
        self._cache_dirty = True
        self._recompute_coeff()

    def _ensure_stats_initialized(self, feature_dim: int) -> None:
        p = self.degree + 1
        if self._device is None:
            raise RuntimeError("Spectrum forecaster stats device is not configured.")
        if self._gram is None or self._gram.shape != (p, p) or self._gram.device != self._device:
            self._gram = torch.zeros((p, p), device=self._device, dtype=torch.float32)
        if self._rhs is None or self._rhs.shape != (p, feature_dim) or self._rhs.device != self._device:
            self._rhs = torch.zeros((p, feature_dim), device=self._device, dtype=torch.float32)

    def _accumulate_feature(self, time_coord: float, feature_flat: torch.Tensor, *, sign: float) -> None:
        if self._gram is None or self._rhs is None:
            self._ensure_stats_initialized(feature_flat.numel())
        basis_row = self._build_design(
            torch.tensor([float(time_coord)], device=self._device, dtype=torch.float32),
            self.degree,
        ).reshape(-1)
        self._gram.add_(float(sign) * torch.outer(basis_row, basis_row))
        self._rhs.addmm_(basis_row.unsqueeze(1), feature_flat.unsqueeze(0), beta=1.0, alpha=float(sign))

    def _accumulate_entry(self, entry: _HistoryEntry, *, sign: float) -> None:
        feature_flat = entry.feature_flat.to(device=self._device, dtype=torch.float32)
        self._accumulate_feature(entry.time_coord, feature_flat, sign=sign)

    def _archive_feature_for_history(self, feature: torch.Tensor) -> torch.Tensor:
        flat = feature.reshape(-1)
        if flat.device.type == "cpu":
            return flat.clone()
        use_pinned_copy = flat.device.type == "cuda"
        archived = torch.empty(flat.shape, device="cpu", dtype=flat.dtype, pin_memory=use_pinned_copy)
        archived.copy_(flat, non_blocking=use_pinned_copy)
        return archived

    def _rebuild_stats(self) -> None:
        feature_dim = self._history[0].feature_flat.numel() if self._history else 0
        p = self.degree + 1
        if self._device is None:
            self._gram = None
            self._rhs = None
            return
        self._gram = torch.zeros((p, p), device=self._device, dtype=torch.float32)
        self._rhs = torch.zeros((p, feature_dim), device=self._device, dtype=torch.float32) if feature_dim > 0 else None
        for entry in self._history:
            self._accumulate_entry(entry, sign=1.0)

    @staticmethod
    def _resolve_predict_dtype(dtype: torch.dtype) -> torch.dtype:
        return dtype if torch.is_floating_point(torch.empty((), dtype=dtype)) else torch.float32

    def _mirror_feature_for_prediction(self, feature: torch.Tensor) -> torch.Tensor:
        if self._predict_device is None or self._predict_dtype is None:
            raise RuntimeError("Spectrum forecaster prediction device is not configured.")
        return feature.reshape(-1).to(device=self._predict_device, dtype=self._predict_dtype, copy=True)

    def _refresh_prediction_mirrors(self) -> None:
        self._previous_feature_flat_device = None
        self._previous_time_coord = None
        self._latest_feature_flat_device = None
        self._latest_time_coord = None
        self._coeff_device = None
        if (
            self._predict_device is None
            or self._predict_dtype is None
            or not self._history
            or not self._linear_mirrors_enabled
        ):
            return
        if len(self._history) >= 2:
            previous = self._history[-2]
            self._previous_feature_flat_device = previous.feature_flat.to(
                device=self._predict_device, dtype=self._predict_dtype
            )
            self._previous_time_coord = previous.time_coord
        latest = self._history[-1]
        self._latest_feature_flat_device = latest.feature_flat.to(device=self._predict_device, dtype=self._predict_dtype)
        self._latest_time_coord = latest.time_coord

    def _sync_linear_prediction_mirrors(
        self,
        entry: _HistoryEntry,
        *,
        feat: torch.Tensor,
        linear_mirrors_enabled: bool,
        force_rebuild: bool,
    ) -> None:
        self._linear_mirrors_enabled = linear_mirrors_enabled
        if not self._linear_mirrors_enabled or not self._history:
            self._previous_feature_flat_device = None
            self._previous_time_coord = None
            self._latest_feature_flat_device = None
            self._latest_time_coord = None
            return
        if force_rebuild or self._latest_feature_flat_device is None:
            self._refresh_prediction_mirrors()
            return

        self._previous_feature_flat_device = self._latest_feature_flat_device
        self._previous_time_coord = self._latest_time_coord
        self._latest_feature_flat_device = self._mirror_feature_for_prediction(feat)
        self._latest_time_coord = entry.time_coord

    def _build_design(self, coords: torch.Tensor, degree: int) -> torch.Tensor:
        coords = coords.reshape(-1, 1).to(torch.float32)
        cols = [torch.ones((coords.shape[0], 1), device=coords.device, dtype=torch.float32)]
        if degree >= 1:
            cols.append(coords)
            for _ in range(2, degree + 1):
                cols.append(2.0 * coords * cols[-1] - cols[-2])
        return torch.cat(cols[: degree + 1], dim=1)

    def _solve(self, design: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        p = design.shape[1]
        lhs = design.transpose(0, 1) @ design
        if self.ridge_lambda > 0.0:
            lhs = lhs + self.ridge_lambda * torch.eye(p, device=design.device, dtype=design.dtype)
        rhs = design.transpose(0, 1) @ features
        try:
            chol = torch.linalg.cholesky(lhs)
        except RuntimeError:
            diag_mean = lhs.diag().mean() if lhs.numel() else torch.tensor(1.0, device=lhs.device)
            jitter = max(float(diag_mean.item()) * 1e-6, 1e-8)
            chol = torch.linalg.cholesky(lhs + jitter * torch.eye(p, device=lhs.device, dtype=lhs.dtype))
        return torch.cholesky_solve(rhs, chol)

    def _recompute_coeff(self) -> None:
        if not self.ready() or self._gram is None or self._rhs is None:
            self._coeff = None
            self._coeff_device = None
            self._cached_degree = None
            self._cache_dirty = True
            return

        degree = self.degree
        lhs = self._gram
        rhs = self._rhs
        if lhs.numel() == 0 or rhs.numel() == 0:
            self._coeff = None
            self._coeff_device = None
            self._cached_degree = None
            self._cache_dirty = True
            return
        if self.ridge_lambda > 0.0:
            lhs = lhs + self.ridge_lambda * torch.eye(degree + 1, device=lhs.device, dtype=lhs.dtype)
        try:
            chol = torch.linalg.cholesky(lhs)
        except RuntimeError:
            diag_mean = lhs.diag().mean() if lhs.numel() else torch.tensor(1.0, device=lhs.device)
            jitter = max(float(diag_mean.item()) * 1e-6, 1e-8)
            chol = torch.linalg.cholesky(lhs + jitter * torch.eye(degree + 1, device=lhs.device, dtype=lhs.dtype))
        self._coeff = torch.cholesky_solve(rhs, chol)
        self._coeff_device = None
        self._cached_degree = degree
        self._cache_dirty = False

    def _ensure_coeff(self) -> tuple[int, torch.Tensor]:
        degree = self.degree
        if not self._cache_dirty and self._coeff is not None and self._cached_degree == degree:
            return degree, self._coeff

        self._recompute_coeff()
        if self._coeff is None or self._cached_degree is None:
            raise RuntimeError("Spectrum forecaster coefficients are not ready yet.")
        return self._cached_degree, self._coeff

    def _ensure_coeff_device(self) -> torch.Tensor:
        if self._coeff is None or self._predict_device is None or self._predict_dtype is None:
            raise RuntimeError("Spectrum forecaster prediction coefficients are not ready yet.")
        if self._coeff_device is None:
            if self._coeff.device == self._predict_device and self._coeff.dtype == self._predict_dtype:
                self._coeff_device = self._coeff
            else:
                self._coeff_device = self._coeff.to(device=self._predict_device, dtype=self._predict_dtype)
        return self._coeff_device

    def _linear_prediction(self, time_coord: float) -> torch.Tensor:
        if self._latest_feature_flat_device is None or self._latest_time_coord is None:
            raise RuntimeError("Spectrum forecaster has no cached feature history.")
        if self._previous_feature_flat_device is None or self._previous_time_coord is None:
            return self._latest_feature_flat_device

        delta_coord = self._latest_time_coord - self._previous_time_coord
        if abs(delta_coord) <= 1e-12:
            return self._latest_feature_flat_device

        k = (float(time_coord) - float(self._latest_time_coord)) / float(delta_coord)
        last_f = self._latest_feature_flat_device
        prev_f = self._previous_feature_flat_device
        return last_f + k * (last_f - prev_f)

    def predict(self, time_coord: float, blend_weight: float) -> torch.Tensor:
        if (
            self._feature_shape is None
            or self._feature_dtype is None
            or self._device is None
            or self._predict_device is None
            or self._predict_dtype is None
            or self._output_device is None
        ):
            raise RuntimeError("Spectrum forecaster has no cached feature history.")
        if not self.ready():
            raise RuntimeError("Spectrum forecaster is not ready yet.")

        degree, _ = self._ensure_coeff()
        coeff_device = self._ensure_coeff_device()

        coord_star = torch.tensor([float(time_coord)], device=self._predict_device, dtype=torch.float32)
        design_star = self._build_design(coord_star, degree).to(dtype=coeff_device.dtype)
        spectral = (design_star @ coeff_device).reshape(self._feature_shape)

        blend = float(blend_weight)
        if blend >= (1.0 - 1e-12):
            out = spectral
        else:
            linear = self._linear_prediction(time_coord).reshape(self._feature_shape)
            out = blend * spectral + (1.0 - blend) * linear
        return out.to(device=self._output_device, dtype=self._feature_dtype)
