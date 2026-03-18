from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(slots=True)
class SpectrumConfig:
    enabled: bool = True
    backend: str = "flux"
    blend_weight: float = 0.50
    degree: int = 4
    ridge_lambda: float = 0.10
    window_size: float = 2.0
    flex_window: float = 0.75
    warmup_steps: int = 5
    tail_actual_steps: int = 3
    max_history: int = 128
    debug: bool = False

    def validate(self) -> "SpectrumConfig":
        if self.backend != "flux":
            raise ValueError(f"Unsupported backend: {self.backend!r}")
        if not (0.0 <= float(self.blend_weight) <= 1.0):
            raise ValueError("blend_weight must be in [0, 1].")
        if int(self.degree) < 1:
            raise ValueError("degree must be >= 1.")
        if float(self.ridge_lambda) < 0.0:
            raise ValueError("ridge_lambda must be >= 0.")
        if float(self.window_size) < 1.0:
            raise ValueError("window_size must be >= 1.")
        if float(self.flex_window) < 0.0:
            raise ValueError("flex_window must be >= 0.")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0.")
        if int(self.tail_actual_steps) < 0:
            raise ValueError("tail_actual_steps must be >= 0.")
        if int(self.max_history) < max(8, int(self.degree) + 1):
            raise ValueError("max_history must be at least max(8, degree + 1).")
        return self

    def to_dict(self) -> dict:
        return asdict(self)
