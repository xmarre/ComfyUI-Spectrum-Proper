# ComfyUI-Spectrum-Proper

Faithful **ComfyUI FLUX** port of **Spectrum** from *Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*.

This repo is intentionally narrow in scope: it implements the **FLUX path properly** instead of trying to be a half-faithful generic accelerator for every backend.

## What this node does

`Spectrum Apply Flux` patches the native ComfyUI FLUX diffusion model on the **MODEL** path and applies Spectrum-style forecasting to the **final hidden image feature right before `final_layer`**.

That matches the official Spectrum FLUX integration much more closely than forecasting the final denoised output tensor.

In practice the flow is:

1. run a real FLUX forward on selected steps
2. cache the final pre-head hidden feature
3. fit a small Chebyshev ridge regressor online over step index
4. forecast future pre-head features on skipped steps
5. still apply the original FLUX `final_layer` for the current conditioning

## Why this repo exists

The currently circulating ComfyUI Spectrum ports are useful experiments, but they miss important invariants from the paper and the official code.

The main problems I corrected here are:

- **Wrong prediction target** in one SDXL-style port: it forecasts the whole UNet output instead of the final hidden feature at the model-specific integration point.
- **Runtime leakage across model clones** in one FLUX port: it closes over a specific runtime object when monkey-patching the shared inner FLUX model.
- **Hard-coded 50-step normalization** without adapting to the actual detected run length.
- **Heuristic pass resets** based on timestep direction only, which are brittle in real ComfyUI workflows.
- **No clean fallback path** for models that share the same patched inner FLUX object but are not actually using Spectrum.

This implementation installs a **generic wrapper once** on the shared FLUX inner model and looks up the active Spectrum runtime from `transformer_options` per call. That avoids ghost patching across clones and preserves normal behavior for non-Spectrum models.

## Current scope

Supported:

- native **ComfyUI FLUX** models
- LoRAs on the normal model path
- standard `transformer_options` patch chains
- standard FLUX control residuals

Not included:

- SDXL
- SD3.5
- video backends

That omission is deliberate. A proper SDXL port in ComfyUI needs either a stable last-block hook in the native UNet path or a maintained fork of the UNet forward. Shipping a brittle pseudo-port would be worse than not shipping one.

## Installation

Copy this folder into:

```text
ComfyUI/custom_nodes/ComfyUI-Spectrum-Proper
```

Restart ComfyUI.

No extra Python dependencies are required beyond what ComfyUI already provides.

## Node

### Spectrum Apply Flux

**Input:** `MODEL`

**Output:** `MODEL`

Place it on the FLUX model line:

```text
UNETLoader / CheckpointLoader -> LoRA stack -> Spectrum Apply Flux -> CFGGuider / sampler
```

Recommended placement:

- after model loading and LoRA application
- before guider/sampler nodes

## Parameters

### `blend_weight`
Blend between linear local extrapolation and Chebyshev spectral prediction.

- `1.0` = pure spectral predictor
- `0.0` = pure local linear predictor
- recommended default: `0.5`

The official repo notes that a convex blend improves robustness outside the strict paper setting.

### `degree`
Chebyshev degree `m`.

Recommended default: `4`

### `ridge_lambda`
Ridge regularization `λ` for the coefficient fit.

Recommended default: `0.1`

### `window_size`
Initial interval size before a real forward is required again.

Recommended default: `2.0`

### `flex_window`
How much the interval grows after each post-warmup real forward.

This is the ComfyUI-facing equivalent of the adaptive schedule slope used in the official repo.

- `0.75` = paper-style moderate speedup
- `3.0` = more aggressive speedup

### `warmup_steps`
Number of initial real forwards before forecasting is allowed.

Recommended default: `5`

### `max_history`
Cap for cached real-forward feature points used for the fit.

This is an implementation guard, not a paper hyperparameter. With standard FLUX schedules it is usually far above the number of actual cached points anyway.

### `debug`
Enables lightweight logging during patch install.

## Recommended settings

### Safer / closer to the paper’s moderate setting

- `blend_weight = 0.50`
- `degree = 4`
- `ridge_lambda = 0.10`
- `window_size = 2.0`
- `flex_window = 0.75`
- `warmup_steps = 5`

### More aggressive

- `blend_weight = 0.75`
- `degree = 4`
- `ridge_lambda = 0.10`
- `window_size = 2.0`
- `flex_window = 3.0`
- `warmup_steps = 5`

## Design notes

### 1. Forecast target is the final hidden FLUX image feature
This repo caches and forecasts the hidden image tokens **after the single-stream blocks and before `final_layer`**.

That is the important architectural choice. Forecasting the final model output directly is less faithful to the official FLUX integration and tends to be less stable.

### 2. Runtime state is per patched model, not per globally monkey-patched inner model
ComfyUI model clones often share the same underlying diffusion model object. If you close over a runtime object when replacing `forward_orig`, the state can leak between clones.

This repo avoids that by:

- patching the inner FLUX model only once
- storing the active runtime in each cloned model’s `transformer_options`
- looking up the runtime dynamically on every call
- falling back to the original `forward_orig` when Spectrum is not active

### 3. Step normalization uses detected schedule length
The paper and official code mostly benchmark 50-step runs. ComfyUI users do not.

This repo normalizes the Chebyshev basis against the detected schedule length from `sample_sigmas` instead of hard-coding 50 steps.

## Known limitations

- This repo currently targets **native ComfyUI FLUX only**.
- It depends on current ComfyUI FLUX internals staying broadly compatible with the present `forward_orig` signature.
- It is designed to coexist with standard transformer patch chains, but it is **not guaranteed** to compose with other custom nodes that also replace FLUX `forward_orig` directly.
- The scheduler is faithful to the official adaptive-window strategy, but one safety approximation is added: forecasting is held back until enough real points exist to fit the chosen Chebyshev degree.
- No claims are made here about exact paper speedups inside arbitrary ComfyUI workflows. Sampler choice, guidance path, ControlNet usage, resolution, and other wrappers all affect real wall-clock results.

## Validation / smoke test

Outside ComfyUI, you can at least validate the scheduler and forecaster math:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Spectrum-Proper
python tests/smoke_runtime.py
```

Expected output:

```text
ok
```

## Repo structure

```text
ComfyUI-Spectrum-Proper/
├── __init__.py
├── nodes.py
├── pyproject.toml
├── LICENSE
├── README.md
├── comfyui_spectrum/
│   ├── __init__.py
│   ├── config.py
│   ├── forecast.py
│   ├── flux.py
│   └── runtime.py
└── tests/
    └── smoke_runtime.py
```

## Credits

- Spectrum paper and official code by Jiaqi Han et al.
- ComfyUI FLUX integration details adapted against current native ComfyUI FLUX internals

## License

GPL-3.0-or-later. This is the safest choice because parts of the FLUX forward-path integration are adapted against ComfyUI core internals.
