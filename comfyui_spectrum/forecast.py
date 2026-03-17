from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(slots=True)
class _HistoryEntry:
    time_coord: float
    feature: torch.Tensor


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

    def configure(self, degree: int, ridge_lambda: float, max_history: int) -> None:
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self.max_history = int(max_history)

    def ready(self, min_points: Optional[int] = None) -> bool:
        needed = max(2, int(min_points) if min_points is not None else self.degree + 1)
        return len(self._history) >= needed

    def update(self, time_coord: float, feature: torch.Tensor) -> None:
        feat = feature.detach()
        if self._feature_shape is None:
            self._feature_shape = feat.shape
            self._feature_dtype = feat.dtype
            self._device = feat.device
        elif feat.shape != self._feature_shape:
            raise ValueError(
                f"Spectrum feature shape changed from {tuple(self._feature_shape)} to {tuple(feat.shape)}."
            )

        self._history.append(_HistoryEntry(float(time_coord), feat))
        if len(self._history) > self.max_history:
            self._history.pop(0)

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

    def _linear_prediction(self, time_coord: float) -> torch.Tensor:
        last = self._history[-1]
        if len(self._history) < 2:
            return last.feature.to(torch.float32)

        prev = self._history[-2]
        delta_coord = last.time_coord - prev.time_coord
        if abs(delta_coord) <= 1e-12:
            return last.feature.to(torch.float32)

        k = (float(time_coord) - float(last.time_coord)) / float(delta_coord)
        last_f = last.feature.to(torch.float32)
        prev_f = prev.feature.to(torch.float32)
        return last_f + k * (last_f - prev_f)

    def predict(self, time_coord: float, blend_weight: float) -> torch.Tensor:
        if self._feature_shape is None or self._feature_dtype is None or self._device is None:
            raise RuntimeError("Spectrum forecaster has no cached feature history.")
        if not self.ready():
            raise RuntimeError("Spectrum forecaster is not ready yet.")

        degree = min(self.degree, len(self._history) - 1)
        coords = torch.tensor([entry.time_coord for entry in self._history], device=self._device, dtype=torch.float32)
        features = torch.stack(
            [entry.feature.reshape(-1).to(torch.float32) for entry in self._history],
            dim=0,
        )

        design = self._build_design(coords, degree)
        coeff = self._solve(design, features)

        coord_star = torch.tensor([float(time_coord)], device=self._device, dtype=torch.float32)
        design_star = self._build_design(coord_star, degree)
        spectral = (design_star @ coeff).reshape(self._feature_shape)

        linear = self._linear_prediction(time_coord).reshape(self._feature_shape)
        out = float(blend_weight) * spectral + (1.0 - float(blend_weight)) * linear
        return out.to(dtype=self._feature_dtype)
