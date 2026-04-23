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
    _supports_solver_step_tracking,
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


def test_supported_single_eval_sampler_names_include_supported_variants() -> None:
    assert "sample_euler" in _SUPPORTED_SINGLE_EVAL_SAMPLERS
    assert "sample_euler_ancestral" in _SUPPORTED_SINGLE_EVAL_SAMPLERS
    assert "sample_euler_flow" in _SUPPORTED_SINGLE_EVAL_SAMPLERS
    assert "sample_lcm" in _SUPPORTED_SINGLE_EVAL_SAMPLERS
    assert "sample_dpmpp_2m_sde" in _SUPPORTED_SINGLE_EVAL_SAMPLERS
    assert "sample_dpmpp_3m_sde" in _SUPPORTED_SINGLE_EVAL_SAMPLERS
    assert "euler_flow" in _SUPPORTED_SINGLE_EVAL_SAMPLERS
    assert "Flux2JiTSamplerImpl" in _SUPPORTED_SINGLE_EVAL_SAMPLERS

    class _FunctionSampler:
        def __init__(self, fn):
            self.sampler_function = fn

    def sample_euler_ancestral(*args, **kwargs):
        raise NotImplementedError

    def sample_euler_flow(*args, **kwargs):
        raise NotImplementedError

    def sample_lcm(*args, **kwargs):
        raise NotImplementedError

    def sample_dpmpp_2m_sde(*args, **kwargs):
        raise NotImplementedError

    def sample_dpmpp_3m_sde(*args, **kwargs):
        raise NotImplementedError

    EulerFlowSampler = type("euler_flow", (), {})
    Flux2JiTSamplerImpl = type("Flux2JiTSamplerImpl", (), {})

    assert _supports_solver_step_tracking(_FunctionSampler(sample_euler_ancestral)) is True
    assert _supports_solver_step_tracking(_FunctionSampler(sample_euler_flow)) is True
    assert _supports_solver_step_tracking(_FunctionSampler(sample_lcm)) is True
    assert _supports_solver_step_tracking(_FunctionSampler(sample_dpmpp_2m_sde)) is True
    assert _supports_solver_step_tracking(_FunctionSampler(sample_dpmpp_3m_sde)) is True
    assert _supports_solver_step_tracking(EulerFlowSampler()) is True
    assert _supports_solver_step_tracking(Flux2JiTSamplerImpl()) is True


def test_forecaster_recomputes_coeff_on_update_not_predict() -> None:
    class CountingForecaster(ChebyshevSpectrumForecaster):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recompute_calls = 0

        def _recompute_coeff(self) -> None:
            self.recompute_calls += 1
            return super()._recompute_coeff()

    forecaster = CountingForecaster(degree=4, ridge_lambda=0.1, max_history=8)
    for idx in range(5):
        forecaster.update(float(idx), torch.randn(1, 8, 4))

    assert forecaster._history[0].feature_flat.device.type == "cpu"

    recompute_calls_before_predict = forecaster.recompute_calls
    first = forecaster.predict(5.0, blend_weight=0.5)
    second = forecaster.predict(5.5, blend_weight=0.5)
    assert first.shape == (1, 8, 4)
    assert second.shape == (1, 8, 4)
    assert forecaster.recompute_calls == recompute_calls_before_predict

    recompute_calls_before_update = forecaster.recompute_calls
    forecaster.update(6.0, torch.randn(1, 8, 4))
    assert forecaster.recompute_calls == recompute_calls_before_update + 1


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
            branch_signature=(("cond_or_uncond", (0, 1)), ("uuids", ("u0", "u1"))),
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
        branch_signature=(("cond_or_uncond", (0, 1)), ("uuids", ("u0", "u1"))),
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


def test_batch_split_falls_back_to_actual_without_disabling_run() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        first_id = runtime.register_model_hook_call(
            run_id, step_id, expected_shape=(1, 8, 4), branch_signature=(("cond_or_uncond", (1,)), ("uuids", ("u1",)))
        )
        runtime.observe_actual_feature(run_id, step_id, torch.ones(1, 8, 4) * 10.0, call_id=first_id)
        second_id = runtime.register_model_hook_call(
            run_id, step_id, expected_shape=(1, 8, 4), branch_signature=(("cond_or_uncond", (0,)), ("uuids", ("u0",)))
        )
        runtime.observe_actual_feature(run_id, step_id, torch.ones(1, 8, 4) * 20.0, call_id=second_id)
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    assert decision["actual_forward"] is False
    call_id = runtime.register_model_hook_call(
        run_id, 5, expected_shape=(2, 8, 4), branch_signature=(("cond_or_uncond", (0, 1)), ("uuids", ("u0", "u1")))
    )
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(2, 8, 4), call_id=call_id)
    assert predicted is not None
    assert predicted.shape == (2, 8, 4)
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=True)
    runtime.end_run(run_id)


def test_nonbatch_shape_mismatch_disables_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        runtime.begin_solver_step(
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
    call_id = runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 9, 4))
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(1, 9, 4), call_id=call_id)
    assert predicted is None
    assert runtime.stats.forecast_disabled is True
    assert runtime.stats.disable_reason == "predicted feature shape did not match the current solver-step input"
    runtime.observe_actual_feature(run_id, 5, torch.randn(1, 9, 4), call_id=call_id)
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)
    runtime.end_run(run_id)


def test_multiple_hook_calls_are_aggregated_on_actual_steps() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        first_id = runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4))
        runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4), call_id=first_id)
        second_id = runtime.register_model_hook_call(run_id, step_id, expected_shape=(1, 8, 4))
        runtime.observe_actual_feature(run_id, step_id, torch.randn(1, 8, 4), call_id=second_id)
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    call_id = runtime.register_model_hook_call(run_id, 5, expected_shape=(2, 8, 4))
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(2, 8, 4), call_id=call_id)
    assert predicted is not None
    assert predicted.shape == (2, 8, 4)
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=True)
    runtime.end_run(run_id)


def test_split_forecast_step_uses_spectrum_across_subcalls() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        first_id = runtime.register_model_hook_call(
            run_id, step_id, expected_shape=(1, 8, 4), branch_signature=(("cond_or_uncond", (1,)), ("uuids", ("u1",)))
        )
        runtime.observe_actual_feature(run_id, step_id, torch.ones(1, 8, 4) * 10.0, call_id=first_id)
        second_id = runtime.register_model_hook_call(
            run_id, step_id, expected_shape=(1, 8, 4), branch_signature=(("cond_or_uncond", (0,)), ("uuids", ("u0",)))
        )
        runtime.observe_actual_feature(run_id, step_id, torch.ones(1, 8, 4) * 20.0, call_id=second_id)
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    assert decision["actual_forward"] is False

    first_id = runtime.register_model_hook_call(
        run_id, 5, expected_shape=(1, 8, 4), branch_signature=(("cond_or_uncond", (0,)), ("uuids", ("u0",)))
    )
    first_pred = runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4), call_id=first_id)
    assert first_pred is not None
    assert first_pred.shape == (1, 8, 4)

    second_id = runtime.register_model_hook_call(
        run_id, 5, expected_shape=(1, 8, 4), branch_signature=(("cond_or_uncond", (1,)), ("uuids", ("u1",)))
    )
    second_pred = runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4), call_id=second_id)
    assert second_pred is not None
    assert second_pred.shape == (1, 8, 4)

    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=True)
    assert runtime.stats.forecast_disabled is False
    assert runtime.stats.forecasted_count == 1
    runtime.end_run(run_id)


def test_split_forecast_step_without_layout_labels_falls_back_to_actual() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        call_id = runtime.register_model_hook_call(run_id, step_id, expected_shape=(2, 8, 4), branch_signature=None)
        runtime.observe_actual_feature(run_id, step_id, torch.randn(2, 8, 4), call_id=call_id)
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    assert decision["actual_forward"] is False
    first_id = runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4), branch_signature=None)
    assert runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4), call_id=first_id) is None
    runtime.observe_actual_feature(run_id, 5, torch.randn(1, 8, 4), call_id=first_id)
    second_id = runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4), branch_signature=None)
    assert runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4), call_id=second_id) is None
    runtime.observe_actual_feature(run_id, 5, torch.randn(1, 8, 4), call_id=second_id)
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)
    assert runtime.stats.forecast_disabled is False
    assert runtime.stats.forecasted_count == 0
    runtime.end_run(run_id)



def test_chunk_level_cond_or_uncond_and_uuid_labels_expand_to_rows_for_split_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        runtime.begin_solver_step(run_id, step_id, runtime.time_coord_for_step(step_id), total_steps)
        call_id = runtime.register_model_hook_call(
            run_id,
            step_id,
            expected_shape=(4, 8, 4),
            branch_signature=(("cond_or_uncond", (1, 0)), ("uuids", ("u1", "u0"))),
        )
        runtime.observe_actual_feature(run_id, step_id, torch.randn(4, 8, 4), call_id=call_id)
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(run_id, 5, runtime.time_coord_for_step(5), total_steps)
    assert decision["actual_forward"] is False
    first_id = runtime.register_model_hook_call(
        run_id, 5, expected_shape=(2, 8, 4), branch_signature=(("cond_or_uncond", (0,)), ("uuids", ("u0",))),
    )
    assert runtime.predict_feature(run_id, 5, expected_shape=(2, 8, 4), call_id=first_id) is not None
    second_id = runtime.register_model_hook_call(
        run_id, 5, expected_shape=(2, 8, 4), branch_signature=(("cond_or_uncond", (1,)), ("uuids", ("u1",))),
    )
    assert runtime.predict_feature(run_id, 5, expected_shape=(2, 8, 4), call_id=second_id) is not None
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=True)
    assert runtime.stats.forecasted_count == 1
    runtime.end_run(run_id)

def test_duplicate_batch_labels_can_be_reordered_for_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        first_id = runtime.register_model_hook_call(
            run_id, step_id, expected_shape=(2, 8, 4), branch_signature=(("cond_or_uncond", (1, 1)), ("uuids", ("u1a", "u1b")))
        )
        runtime.observe_actual_feature(run_id, step_id, torch.ones(2, 8, 4) * 10.0, call_id=first_id)
        second_id = runtime.register_model_hook_call(
            run_id, step_id, expected_shape=(2, 8, 4), branch_signature=(("cond_or_uncond", (0, 0)), ("uuids", ("u0a", "u0b")))
        )
        runtime.observe_actual_feature(run_id, step_id, torch.ones(2, 8, 4) * 20.0, call_id=second_id)
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    call_id = runtime.register_model_hook_call(
        run_id, 5, expected_shape=(4, 8, 4), branch_signature=(("cond_or_uncond", (0, 0, 1, 1)), ("uuids", ("u0a", "u0b", "u1a", "u1b")))
    )
    predicted = runtime.predict_feature(run_id, 5, expected_shape=(4, 8, 4), call_id=call_id)
    assert predicted is not None
    assert predicted.shape == (4, 8, 4)
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=True)
    runtime.end_run(run_id)


def test_topology_change_disables_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    decision = runtime.begin_solver_step(
        run_id,
        0,
        runtime.time_coord_for_step(0),
        total_steps,
    )
    first_id = runtime.register_model_hook_call(
        run_id, 0, expected_shape=(1, 8, 4), branch_signature=(("hooks_id", 1), ("cond_or_uncond", (0,)), ("uuids", ("u0",)))
    )
    runtime.observe_actual_feature(run_id, 0, torch.randn(1, 8, 4), call_id=first_id)
    second_id = runtime.register_model_hook_call(
        run_id, 0, expected_shape=(1, 8, 4), branch_signature=(("hooks_id", 2), ("cond_or_uncond", (1,)), ("uuids", ("u1",)))
    )
    assert runtime.stats.forecast_disabled is True
    assert runtime.stats.disable_reason == "model-hook branch signature changed within one solver step"
    runtime.observe_actual_feature(run_id, 0, torch.randn(1, 8, 4), call_id=second_id)
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)
    runtime.end_run(run_id)


def test_mixed_batch_layout_presence_disables_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    decision = runtime.begin_solver_step(
        run_id,
        0,
        runtime.time_coord_for_step(0),
        total_steps,
    )
    first_id = runtime.register_model_hook_call(run_id, 0, expected_shape=(1, 8, 4), branch_signature=None)
    runtime.observe_actual_feature(run_id, 0, torch.randn(1, 8, 4), call_id=first_id)
    second_id = runtime.register_model_hook_call(
        run_id, 0, expected_shape=(1, 8, 4), branch_signature=(("cond_or_uncond", (1,)), ("uuids", ("u1",)))
    )
    runtime.observe_actual_feature(run_id, 0, torch.randn(1, 8, 4), call_id=second_id)
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)
    assert runtime.stats.forecast_disabled is True
    assert runtime.stats.disable_reason == "model-hook batch layout changed within one solver step"
    runtime.end_run(run_id)


def test_split_forecast_failure_after_first_slice_still_counts_as_mixed_path() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    for step_id in range(5):
        runtime.begin_solver_step(
            run_id,
            step_id,
            runtime.time_coord_for_step(step_id),
            total_steps,
        )
        first_id = runtime.register_model_hook_call(
            run_id, step_id, expected_shape=(1, 8, 4), branch_signature=(("cond_or_uncond", (1,)), ("uuids", ("u1",)))
        )
        runtime.observe_actual_feature(run_id, step_id, torch.ones(1, 8, 4), call_id=first_id)
        second_id = runtime.register_model_hook_call(
            run_id, step_id, expected_shape=(1, 8, 4), branch_signature=(("cond_or_uncond", (0,)), ("uuids", ("u0",)))
        )
        runtime.observe_actual_feature(run_id, step_id, torch.ones(1, 8, 4), call_id=second_id)
        runtime.finalize_solver_step(run_id, step_id, used_forecast=False)

    decision = runtime.begin_solver_step(
        run_id,
        5,
        runtime.time_coord_for_step(5),
        total_steps,
    )
    first_id = runtime.register_model_hook_call(
        run_id, 5, expected_shape=(1, 8, 4), branch_signature=(("cond_or_uncond", (0,)), ("uuids", ("u0",)))
    )
    assert runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4), call_id=first_id) is not None
    second_id = runtime.register_model_hook_call(run_id, 5, expected_shape=(1, 8, 4), branch_signature=None)
    assert runtime.predict_feature(run_id, 5, expected_shape=(1, 8, 4), call_id=second_id) is None
    runtime.observe_actual_feature(run_id, 5, torch.ones(1, 8, 4), call_id=second_id)
    runtime.finalize_solver_step(decision["run_id"], decision["solver_step_id"], used_forecast=False)
    assert runtime.stats.disable_reason == "solver step mixed forecasted and actual model-hook paths"
    runtime.end_run(run_id)


def test_aborted_solver_step_is_discarded_without_disabling_forecast() -> None:
    runtime = make_runtime()
    sample_sigmas = torch.linspace(1.0, 0.0, 51)
    run_id = runtime.start_run(sample_sigmas, "sample_euler", supports_solver_steps=True)
    total_steps = len(sample_sigmas) - 1

    decision = runtime.begin_solver_step(
        run_id,
        0,
        runtime.time_coord_for_step(0),
        total_steps,
    )
    runtime.abort_solver_step(decision["run_id"], decision["solver_step_id"])
    assert runtime.stats.forecast_disabled is False
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


def test_flux_sampler_contract_allows_supported_single_eval_variants() -> None:
    assert _SUPPORTED_SINGLE_EVAL_SAMPLERS == frozenset(
        {
            "sample_euler",
            "sample_euler_ancestral",
            "sample_euler_flow",
            "sample_lcm",
            "sample_dpmpp_2m_sde",
            "sample_dpmpp_3m_sde",
            "euler_flow",
            "Flux2JiTSamplerImpl",
        }
    )


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
    test_supported_single_eval_sampler_names_include_supported_variants()
    test_forecaster_recomputes_coeff_on_update_not_predict()
    test_solver_step_scheduler()
    test_forecast_fallback_reconciles_bookkeeping()
    test_observe_actual_feature_clears_forecast_latch()
    test_unsupported_sampler_disables_forecast()
    test_batch_split_falls_back_to_actual_without_disabling_run()
    test_nonbatch_shape_mismatch_disables_forecast()
    test_multiple_hook_calls_are_aggregated_on_actual_steps()
    test_split_forecast_step_uses_spectrum_across_subcalls()
    test_split_forecast_step_without_layout_labels_falls_back_to_actual()
    test_chunk_level_cond_or_uncond_and_uuid_labels_expand_to_rows_for_split_forecast()
    test_duplicate_batch_labels_can_be_reordered_for_forecast()
    test_topology_change_disables_forecast()
    test_mixed_batch_layout_presence_disables_forecast()
    test_split_forecast_failure_after_first_slice_still_counts_as_mixed_path()
    test_aborted_solver_step_is_discarded_without_disabling_forecast()
    test_nonuniform_schedule_coords_are_used()
    test_forecaster_respects_nonuniform_coords()
    test_flux_sampler_contract_allows_supported_single_eval_variants()
    test_tail_actual_steps_force_real_forwards()
    test_forecast_feature_is_sanitized_before_fp16_final_layer()
    test_forecast_feature_sanitization_stats_only_report_real_violations()
    print("ok")


if __name__ == "__main__":
    main()
