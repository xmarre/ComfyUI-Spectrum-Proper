from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(slots=True)
class _HistoryEntry:
    time_coord: float
    feature_flat: torch.Tensor
    basis_row: torch.Tensor


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
        self._output_device: Optional[torch.device] = None
        self._coeff: Optional[torch.Tensor] = None
        self._cached_degree: Optional[int] = None
        self._cache_dirty = True
        self._gram: Optional[torch.Tensor] = None
        self._rhs: Optional[torch.Tensor] = None

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
        self._recompute_coeff()

    @property
    def feature_shape(self) -> Optional[torch.Size]:
        return self._feature_shape

    def ready(self, min_points: Optional[int] = None) -> bool:
        needed = max(2, int(min_points) if min_points is not None else self.degree + 1)
        return len(self._history) >= needed

    def update(self, time_coord: float, feature: torch.Tensor) -> None:
        feat = feature.detach()
        if self._feature_shape is None:
            self._feature_shape = feat.shape
            self._feature_dtype = feat.dtype
            self._device = torch.device("cpu")
            self._output_device = feat.device
        elif feat.shape != self._feature_shape:
            raise ValueError(
                f"Spectrum feature shape changed from {tuple(self._feature_shape)} to {tuple(feat.shape)}."
            )

        feature_flat = feat.reshape(-1).to(device="cpu", dtype=torch.float32, copy=True)
        basis_row = self._build_design(
            torch.tensor([float(time_coord)], device="cpu", dtype=torch.float32),
            self.degree,
        ).reshape(-1)
        entry = _HistoryEntry(float(time_coord), feature_flat, basis_row)
        self._history.append(entry)
        self._ensure_stats_initialized(feature_flat.numel())
        self._accumulate_entry(entry, sign=1.0)
        if len(self._history) > self.max_history:
            oldest = self._history.pop(0)
            self._accumulate_entry(oldest, sign=-1.0)
        self._coeff = None
        self._cached_degree = None
        self._cache_dirty = True
        self._recompute_coeff()

    def _ensure_stats_initialized(self, feature_dim: int) -> None:
        p = self.degree + 1
        if self._gram is None or self._gram.shape != (p, p):
            self._gram = torch.zeros((p, p), device="cpu", dtype=torch.float32)
        if self._rhs is None or self._rhs.shape != (p, feature_dim):
            self._rhs = torch.zeros((p, feature_dim), device="cpu", dtype=torch.float32)

    def _accumulate_entry(self, entry: _HistoryEntry, *, sign: float) -> None:
        if self._gram is None or self._rhs is None:
            self._ensure_stats_initialized(entry.feature_flat.numel())
        self._gram.add_(float(sign) * torch.outer(entry.basis_row, entry.basis_row))
        self._rhs.add_(float(sign) * (entry.basis_row.unsqueeze(1) * entry.feature_flat.unsqueeze(0)))

    def _rebuild_stats(self) -> None:
        feature_dim = self._history[0].feature_flat.numel() if self._history else 0
        p = self.degree + 1
        self._gram = torch.zeros((p, p), device="cpu", dtype=torch.float32)
        self._rhs = torch.zeros((p, feature_dim), device="cpu", dtype=torch.float32) if feature_dim > 0 else None
        rebuilt: List[_HistoryEntry] = []
        for entry in self._history:
            basis_row = self._build_design(
                torch.tensor([entry.time_coord], device="cpu", dtype=torch.float32),
                self.degree,
            ).reshape(-1)
            rebuilt_entry = _HistoryEntry(entry.time_coord, entry.feature_flat, basis_row)
            rebuilt.append(rebuilt_entry)
            self._accumulate_entry(rebuilt_entry, sign=1.0)
        self._history = rebuilt

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
            self._cached_degree = None
            self._cache_dirty = True
            return

        degree = self.degree
        lhs = self._gram
        rhs = self._rhs
        if lhs.numel() == 0 or rhs.numel() == 0:
            self._coeff = None
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

    def _linear_prediction(self, time_coord: float) -> torch.Tensor:
        last = self._history[-1]
        if len(self._history) < 2:
            return last.feature_flat

        prev = self._history[-2]
        delta_coord = last.time_coord - prev.time_coord
        if abs(delta_coord) <= 1e-12:
            return last.feature_flat

        k = (float(time_coord) - float(last.time_coord)) / float(delta_coord)
        last_f = last.feature_flat
        prev_f = prev.feature_flat
        return last_f + k * (last_f - prev_f)

    def predict(self, time_coord: float, blend_weight: float) -> torch.Tensor:
        if (
            self._feature_shape is None
            or self._feature_dtype is None
            or self._device is None
            or self._output_device is None
        ):
            raise RuntimeError("Spectrum forecaster has no cached feature history.")
        if not self.ready():
            raise RuntimeError("Spectrum forecaster is not ready yet.")

        degree, coeff = self._ensure_coeff()

        coord_star = torch.tensor([float(time_coord)], device=self._device, dtype=torch.float32)
        design_star = self._build_design(coord_star, degree)
        spectral = (design_star @ coeff).reshape(self._feature_shape)

        linear = self._linear_prediction(time_coord).reshape(self._feature_shape)
        out = float(blend_weight) * spectral + (1.0 - float(blend_weight)) * linear
        return out.to(device=self._output_device, dtype=self._feature_dtype)
