from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyui_spectrum.config import SpectrumConfig
from comfyui_spectrum.forecast import ChebyshevSpectrumForecaster
from comfyui_spectrum.flux import (
    _SUPPORTED_SINGLE_EVAL_SAMPLERS,
    _forecast_feature_sanitization_stats,
    _sanitize_forecast_feature_for_final_layer,
)
from comfyui_spectrum.runtime import SpectrumRuntime


def make_runtime(**overrides) -> SpectrumRuntime:
    cfg_kwargs = {
        "blend_weight": 0.5,
        "degree": 4,
        "ridge_lambda": 0.1,
        "window_size": 2.0,
        "flex_window": 0.75,
        "warmup_steps": 5,
        "tail_actual_steps": 3,
        "max_history": 128,
    }
    cfg_kwargs.update(overrides)
    cfg = SpectrumConfig(**cfg_kwargs).validate()
    return SpectrumRuntime(cfg)


def test_solver_step_scheduler() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        assert decision["actual_forward"] is True
        runtime.register_model_hook_call(
            run_id,
            step_id,
            expected_shape=(1, 8, 4),
            branch_signature=(("cond_or_uncond", (0, 1)),),
        )
        runtime.observe_actual_feature(decision["run_id"], decision["solver_step_id"], torch.randn(1, 8, 4))
        runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    runtime.register_model_hook_call(
        run_id,
        5,
        expected_shape=(1, 8, 4),
        branch_signature=(("cond_or_uncond", (0, 1)),),
    )
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4))
    assert predicted is not None
    assert predicted.shape == (1, 8, 4)
    runtime.finalize_solver_step(run_id, 5, used_forecast=not decision["actual_forward"])
    runtime.end_run(run_id)


def test_forecast_fallback_reconciles_bookkeeping() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4))
        runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4))
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    assert decision["actual_forward"] is False
    before_window = runtime.curr_ws

    runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4))
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(2, 8, 4))
    assert predicted is None
    runtime.observe_actual_feature(run_id, 5, torch.randn(1, 8, 4))
    runtime.finalize_solver_step(run_id, 5, used_forecast=False)

    assert runtime.stats.forecasted_count == 0
    assert runtime.stats.actual_forward_count == 6
    assert runtime.num_consecutive_cached_steps == 0
    assert decision["actual_forward"] is True
    assert runtime.curr_ws == before_window
    runtime.end_run(run_id)


def test_observe_actual_feature_clears_forecast_latch() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4))
        runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4))
        runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4))
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4))
    assert predicted is not None
    assert runtime.step_used_forecast(run_id, 5) is True
    runtime.observe_actual_feature(run_id, 5, torch.randn(1, 8, 4))
    assert runtime.step_used_forecast(run_id, 5) is False
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)
    runtime.end_run(run_id)


def test_unsupported_sampler_disables_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_heun", supports_solver_steps=False)
    decision = runtime.begin_solver_step(
        run_id,
        0,
        runtime.time_coord_for_step(0),
        len(sample_sigmas) - 1,
    )
    assert decision["actual_forward"] is True
    assert runtime.stats.forecast_disabled is True
    assert runtime.stats.disable_reason == "sampler 'sample_heun' does not expose one predict_noise call per solver step"
    runtime.register_model_hook_call(run_id, 0, expected_shape=(1, 8, 4))
    runtime.observe_actual_feature(run_id, 0, torch.randn(1, 8, 4))
    runtime.finalize_solver_step(run_id, 0, used_forecast=False)
    runtime.end_run(run_id)


def test_inconsistent_hook_shape_disables_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4))
        runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4))
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4))
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(2, 8, 4))
    assert predicted is None
    assert runtime.stats.forecast_disabled is True
    assert runtime.stats.disable_reason == "predicted feature shape did not match the current solver-step input"
    runtime.observe_actual_feature(run_id, 5, torch.randn(1, 8, 4))
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)
    runtime.end_run(run_id)


def test_multiple_hook_calls_disable_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4))
        runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4))
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4))
    runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4))
    assert runtime.stats.forecast_disabled is True
    assert runtime.stats.disable_reason == "multiple model-hook calls observed within one solver step"
    runtime.observe_actual_feature(run_id, 5, torch.randn(1, 8, 4))
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)
    runtime.end_run(run_id)


def test_nonuniform_schedule_coords_are_used() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.tensor([10.0, 9.0, 1.0, 0.0])
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)

    coords = [runtime.time_coord_for_step(i) for i in range(3)]
    assert math.isclose(coords[0], -1.0)
    assert math.isclose(coords[2], 1.0)
    assert math.isclose(coords[1], -7.0 / 9.0)
    assert not math.isclose(coords[1] - coords[0], coords[2] - coords[1])

    runtime.end_run(run_id)


def test_forecaster_respects_nonuniform_coords() -> None:
    forecaster = ChebyshevSpectrumForecaster(degree=1, ridge_lambda=0.1, max_history=16)
    forecaster.update(-1.0, torch.tensor([10.0]))
    forecaster.update(-7.0 / 9.0, torch.tensor([9.0]))

    pred = forecaster.predict(time_coord=1.0, blend_weight=0.0)
    assert torch.allclose(pred, torch.tensor([1.0]), atol=1e-5)


def test_flux_sampler_contract_only_allows_euler() -> None:
    assert _SUPPORTED_SINGLE_EVAL_SAMPLERS == frozenset({"sample_euler"})


def test_tail_actual_steps_force_real_forwards() -> None:
    runtime = make_runtime(
        degree=1,
        ridge_lambda=0.1,
        window_size=2.0,
        flex_window=0.75,
        warmup_steps=2,
        tail_actual_steps=3,
        max_history=16,
    )
    sample_sigmas = torch.linspace(1.0, 0.0, 9)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(total_steps):
        decision = runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4))

        if step_id >= total_steps - 3:
            assert decision["actual_forward"] is True
            assert runtime.predict_feature(run_id, step_id, expected_shape=(1, 8, 4)) is None
            runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4))
            runtime.finalize_solver_step(run_id, step_id, used_forecast=False)
            continue

        if step_id < 2:
            assert decision["actual_forward"] is True
            runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4))
            runtime.finalize_solver_step(run_id, step_id, used_forecast=False)
            continue

        if decision["actual_forward"]:
            runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4))
            runtime.finalize_solver_step(run_id, step_id, used_forecast=False)
        else:
            assert runtime.predict_feature(run_id, step_id, expected_shape=(1, 8, 4)) is not None
            runtime.finalize_solver_step(run_id, step_id, used_forecast=True)

    assert runtime.stats.forecasted_count > 0
    assert runtime.stats.actual_forward_count == 6
    runtime.end_run(run_id)


def test_forecast_feature_is_sanitized_before_fp16_final_layer() -> None:
    feature = torch.tensor([float("nan"), float("inf"), float("-inf"), 70000.0, -70000.0, 123.5])
    sanitized = _sanitize_forecast_feature_for_final_layer(feature, torch.float16)

    assert sanitized.dtype == torch.float16
    assert torch.isfinite(sanitized).all()
    assert sanitized.tolist() == [0.0, 65504.0, -65504.0, 65504.0, -65504.0, 123.5]


def test_forecast_feature_sanitization_stats_only_report_real_violations() -> None:
    assert _forecast_feature_sanitization_stats(torch.tensor([1.0, 2.0]), torch.float16) is None

    stats = _forecast_feature_sanitization_stats(
        torch.tensor([float("nan"), float("inf"), -70000.0, 42.0]),
        torch.float16,
    )
    assert stats is not None
    assert stats["target_dtype"] == "torch.float16"
    assert stats["had_nonfinite"] is True
    assert stats["out_of_range"] is True
    assert stats["before_min"] == -70000.0
    assert stats["before_max"] == 42.0


def main() -> None:
    test_solver_step_scheduler()
    test_forecast_fallback_reconciles_bookkeeping()
    test_observe_actual_feature_clears_forecast_latch()
    test_unsupported_sampler_disables_forecast()
    test_inconsistent_hook_shape_disables_forecast()
    test_multiple_hook_calls_disable_forecast()
    test_nonuniform_schedule_coords_are_used()
    test_forecaster_respects_nonuniform_coords()
    test_flux_sampler_contract_only_allows_euler()
    test_tail_actual_steps_force_real_forwards()
    test_forecast_feature_is_sanitized_before_fp16_final_layer()
    test_forecast_feature_sanitization_stats_only_report_real_violations()
    print("ok")


if __name__ == "__main__":
    main()
