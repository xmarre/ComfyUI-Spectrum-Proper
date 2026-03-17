from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyui_spectrum.config import SpectrumConfig
from comfyui_spectrum.runtime import SpectrumRuntime


def main() -> None:
    cfg = SpectrumConfig(
        blend_weight=0.5,
        degree=4,
        ridge_lambda=0.1,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=5,
        max_history=128,
    ).validate()
    runtime = SpectrumRuntime(cfg)
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    transformer_options = {"sample_sigmas": sample_sigmas}

    # Warm up with five actual features.
    for i in range(5):
        decision = runtime.begin_step(transformer_options, torch.tensor([sample_sigmas[i]]))
        assert decision["actual_forward"] is True
        runtime.forecaster.update(i, torch.randn(1, 8, 4))

    # Scheduler should now be able to forecast.
    decision = runtime.begin_step(transformer_options, torch.tensor([sample_sigmas[5]]))
    assert "actual_forward" in decision
    assert runtime.forecaster.predict(
        step_index=5,
        total_steps=decision["total_steps"],
        blend_weight=cfg.blend_weight,
    ).shape == (1, 8, 4)

    print("ok")


if __name__ == "__main__":
    main()
