from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .config import SpectrumConfig
from .runtime import SpectrumRuntime

LOG = logging.getLogger(__name__)
_SUPPORTED_SINGLE_EVAL_SAMPLERS = frozenset(
    {
        "sample_euler",
        "sample_euler_ancestral",
    }
)


def _clone_model(model: Any) -> Any:
    return model.clone() if hasattr(model, "clone") else model


def _ensure_model_options(model: Any) -> Dict[str, Any]:
    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    return model.model_options


def _ensure_transformer_options(model: Any) -> Dict[str, Any]:
    options = _ensure_model_options(model)
    if "transformer_options" not in options or options["transformer_options"] is None:
        options["transformer_options"] = {}
    return options["transformer_options"]


def locate_flux_inner_model(model: Any) -> Tuple[Optional[Any], Optional[str]]:
    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "diffusion_model"):
        return outer.diffusion_model, "model.diffusion_model"
    if hasattr(model, "diffusion_model"):
        return model.diffusion_model, "diffusion_model"
    return None, None


def _invert_slices(slices: Sequence[Tuple[int, int]], length: int):
    sorted_slices = sorted(slices)
    result = []
    current = 0
    for start, end in sorted_slices:
        if current < start:
            result.append((current, start))
        current = max(current, end)
    if current < length:
        result.append((current, length))
    return result


def _sampler_name(sampler: Any) -> str:
    fn = getattr(sampler, "sampler_function", None)
    return getattr(fn, "__name__", type(sampler).__name__)


def _build_branch_signature(transformer_options: Dict[str, Any]) -> Optional[tuple[Any, ...]]:
    signature = []
    cond_or_uncond = transformer_options.get("cond_or_uncond")
    if cond_or_uncond is not None:
        try:
            signature.append(("cond_or_uncond", tuple(int(v) for v in cond_or_uncond)))
        except Exception:
            signature.append(("cond_or_uncond", tuple(cond_or_uncond)))

    uuids = transformer_options.get("uuids")
    if uuids is not None:
        signature.append(("uuids_len", len(uuids)))

    if not signature:
        return None
    return tuple(signature)


def _extract_step_context(transformer_options: Dict[str, Any]) -> Optional[tuple[SpectrumRuntime, int, int, bool]]:
    runtime = transformer_options.get("spectrum_runtime")
    run_id = transformer_options.get("spectrum_run_id")
    solver_step_id = transformer_options.get("spectrum_solver_step_id")
    actual_forward = transformer_options.get("spectrum_actual_forward")
    if runtime is None or run_id is None or solver_step_id is None or actual_forward is None:
        return None
    return runtime, int(run_id), int(solver_step_id), bool(actual_forward)


def _runtime_from_model_options(model_options: Dict[str, Any]) -> Optional[SpectrumRuntime]:
    transformer_options = (model_options or {}).get("transformer_options") or {}
    runtime = transformer_options.get("spectrum_runtime")
    if isinstance(runtime, SpectrumRuntime):
        return runtime
    return None


def _copy_model_options_with_step_context(
    model_options: Dict[str, Any], runtime: SpectrumRuntime, decision: Dict[str, Any]
) -> Dict[str, Any]:
    patched_model_options = dict(model_options or {})
    transformer_options = dict(patched_model_options.get("transformer_options") or {})
    patched_model_options["transformer_options"] = transformer_options
    transformer_options["spectrum_runtime"] = runtime
    transformer_options["spectrum_run_id"] = decision["run_id"]
    transformer_options["spectrum_solver_step_id"] = decision["solver_step_id"]
    transformer_options["spectrum_time_coord"] = decision["time_coord"]
    transformer_options["spectrum_actual_forward"] = decision["actual_forward"]
    transformer_options["spectrum_step_finalized"] = False
    return patched_model_options


def _install_sampler_level_wrappers(model: Any, runtime: SpectrumRuntime) -> None:
    options = _ensure_model_options(model)
    wrapper_state = options.get("_spectrum_sampler_wrapper_state")
    if not isinstance(wrapper_state, dict):
        wrapper_state = {}
        options["_spectrum_sampler_wrapper_state"] = wrapper_state
    wrapper_state["runtime"] = runtime
    if options.get("_spectrum_sampler_wrappers_installed", False):
        return

    import comfy.patcher_extension

    def outer_sample_wrapper(
        executor,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
        latent_shapes=None,
    ):
        wrapper_runtime = wrapper_state.get("runtime")
        if not isinstance(wrapper_runtime, SpectrumRuntime):
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        sampler_name = _sampler_name(sampler)
        supports_solver_steps = sampler_name in _SUPPORTED_SINGLE_EVAL_SAMPLERS
        run_id = wrapper_runtime.start_run(sigmas, sampler_name, supports_solver_steps=supports_solver_steps)
        try:
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        finally:
            wrapper_runtime.end_run(run_id)

    def predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
        effective_model_options = model_options or {}
        wrapper_runtime = _runtime_from_model_options(effective_model_options)
        if wrapper_runtime is None:
            wrapper_runtime = wrapper_state.get("runtime")
        if not isinstance(wrapper_runtime, SpectrumRuntime):
            return executor(x, timestep, effective_model_options, seed)
        if wrapper_runtime.active_run_id is None:
            return executor(x, timestep, effective_model_options, seed)
        if not wrapper_runtime.active_run_supports_solver_steps:
            return executor(x, timestep, effective_model_options, seed)

        step_id = wrapper_runtime.next_solver_step_id
        total_steps = wrapper_runtime.num_steps()
        time_coord = wrapper_runtime.time_coord_for_step(step_id)
        decision = wrapper_runtime.begin_solver_step(wrapper_runtime.active_run_id, step_id, time_coord, total_steps)
        patched_model_options = _copy_model_options_with_step_context(
            effective_model_options, wrapper_runtime, decision
        )
        try:
            return executor(x, timestep, patched_model_options, seed)
        finally:
            wrapper_runtime.finalize_solver_step(
                decision["run_id"],
                decision["solver_step_id"],
                used_forecast=wrapper_runtime.step_used_forecast(
                    decision["run_id"], decision["solver_step_id"]
                ),
            )

    comfy.patcher_extension.add_wrapper(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        outer_sample_wrapper,
        options,
        is_model_options=True,
    )
    comfy.patcher_extension.add_wrapper(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        predict_noise_wrapper,
        options,
        is_model_options=True,
    )
    options["_spectrum_sampler_wrappers_installed"] = True


def _install_generic_flux_wrapper(inner: Any) -> None:
    if getattr(inner, "_spectrum_forward_orig_installed", False):
        return

    original_forward_orig = inner.forward_orig
    setattr(inner, "_spectrum_original_forward_orig", original_forward_orig)

    def spectrum_forward_orig(
        self,
        img,
        img_ids,
        txt,
        txt_ids,
        timesteps,
        y,
        guidance=None,
        control=None,
        timestep_zero_index=None,
        transformer_options=None,
        attn_mask=None,
    ):
        options = transformer_options or {}
        runtime = options.get("spectrum_runtime") if isinstance(options, dict) else None
        if runtime is None or not getattr(runtime.cfg, "enabled", False):
            return self._spectrum_original_forward_orig(
                img,
                img_ids,
                txt,
                txt_ids,
                timesteps,
                y,
                guidance=guidance,
                control=control,
                timestep_zero_index=timestep_zero_index,
                transformer_options=options,
                attn_mask=attn_mask,
            )

        return _run_flux_forward_with_spectrum(
            self,
            runtime,
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            y=y,
            guidance=guidance,
            control=control,
            timestep_zero_index=timestep_zero_index,
            transformer_options=options,
            attn_mask=attn_mask,
        )

    inner.forward_orig = spectrum_forward_orig.__get__(inner, type(inner))
    setattr(inner, "_spectrum_forward_orig_installed", True)


def _run_flux_forward_with_spectrum(
    inner: Any,
    runtime: SpectrumRuntime,
    *,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: torch.Tensor,
    y: torch.Tensor,
    guidance: Optional[torch.Tensor],
    control: Optional[dict],
    timestep_zero_index=None,
    transformer_options: Dict[str, Any],
    attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    from comfy.ldm.flux.layers import timestep_embedding

    transformer_options = (transformer_options or {}).copy()
    patches = transformer_options.get("patches", {})
    patches_replace = transformer_options.get("patches_replace", {})
    step_ctx = _extract_step_context(transformer_options)
    actual_forward = True
    run_id: Optional[int] = None
    solver_step_id: Optional[int] = None

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    img = inner.img_in(img)
    vec = inner.time_in(timestep_embedding(timesteps, 256).to(img.dtype))

    if inner.params.guidance_embed and guidance is not None:
        vec = vec + inner.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    if inner.vector_in is not None:
        if y is None:
            y = torch.zeros((img.shape[0], inner.params.vec_in_dim), device=img.device, dtype=img.dtype)
        vec = vec + inner.vector_in(y[:, : inner.params.vec_in_dim])

    if inner.txt_norm is not None:
        txt = inner.txt_norm(txt)
    txt = inner.txt_in(txt)

    if "post_input" in patches:
        for patch in patches["post_input"]:
            out = patch(
                {
                    "img": img,
                    "txt": txt,
                    "img_ids": img_ids,
                    "txt_ids": txt_ids,
                    "transformer_options": transformer_options,
                }
            )
            img = out["img"]
            txt = out["txt"]
            img_ids = out["img_ids"]
            txt_ids = out["txt_ids"]

    # Spectrum forecasts the final image-token feature right before final_layer.
    expected_feature_shape = (img.shape[0], img.shape[1], *img.shape[2:])

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = inner.pe_embedder(ids)
    else:
        pe = None

    vec_orig = vec
    txt_vec = vec
    extra_kwargs = {}
    modulation_dims = None

    if timestep_zero_index is not None:
        modulation_dims = []
        batch = vec.shape[0] // 2
        vec_orig = vec_orig.reshape(2, batch, vec.shape[1]).movedim(0, 1)
        for start, end in _invert_slices(timestep_zero_index, img.shape[1]):
            modulation_dims.append((start, end, 0))
        for start, end in timestep_zero_index:
            modulation_dims.append((start, end, 1))
        extra_kwargs["modulation_dims_img"] = modulation_dims
        txt_vec = vec[:batch]

    if step_ctx is not None:
        _, run_id, solver_step_id, actual_forward = step_ctx
        runtime.register_model_hook_call(
            run_id,
            solver_step_id,
            expected_shape=expected_feature_shape,
            branch_signature=_build_branch_signature(transformer_options),
        )
        if not actual_forward:
            pred_feature = runtime.predict_feature(
                run_id,
                solver_step_id,
                expected_shape=expected_feature_shape,
            )
            if pred_feature is not None:
                final_kwargs = {}
                if modulation_dims is not None:
                    final_kwargs["modulation_dims"] = modulation_dims
                return inner.final_layer(pred_feature.to(img.dtype), vec_orig, **final_kwargs)

    if inner.params.global_modulation:
        vec = (inner.double_stream_modulation_img(vec_orig), inner.double_stream_modulation_txt(txt_vec))

    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(inner.double_blocks)
    transformer_options["block_type"] = "double"

    for block_index, block in enumerate(inner.double_blocks):
        transformer_options["block_index"] = block_index
        if ("double_block", block_index) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(
                    img=args["img"],
                    txt=args["txt"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                    transformer_options=args.get("transformer_options"),
                    **extra_kwargs,
                )
                return out

            out = blocks_replace[("double_block", block_index)](
                {
                    "img": img,
                    "txt": txt,
                    "vec": vec,
                    "pe": pe,
                    "attn_mask": attn_mask,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(
                img=img,
                txt=txt,
                vec=vec,
                pe=pe,
                attn_mask=attn_mask,
                transformer_options=transformer_options,
                **extra_kwargs,
            )

        if control is not None:
            control_i = control.get("input")
            if control_i is not None and block_index < len(control_i):
                add = control_i[block_index]
                if add is not None:
                    img[:, : add.shape[1]] += add

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    img = torch.cat((txt, img), dim=1)

    if inner.params.global_modulation:
        vec, _ = inner.single_stream_modulation(vec_orig)

    extra_kwargs = {}
    if modulation_dims is not None:
        modulation_dims_combined = [
            (0 if start == 0 else start + txt.shape[1], end + txt.shape[1], kind)
            for start, end, kind in modulation_dims
        ]
        extra_kwargs["modulation_dims"] = modulation_dims_combined

    transformer_options["total_blocks"] = len(inner.single_blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [txt.shape[1], img.shape[1]]

    for block_index, block in enumerate(inner.single_blocks):
        transformer_options["block_index"] = block_index
        if ("single_block", block_index) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                    transformer_options=args.get("transformer_options"),
                    **extra_kwargs,
                )
                return out

            out = blocks_replace[("single_block", block_index)](
                {
                    "img": img,
                    "vec": vec,
                    "pe": pe,
                    "attn_mask": attn_mask,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )
            img = out["img"]
        else:
            img = block(
                img,
                vec=vec,
                pe=pe,
                attn_mask=attn_mask,
                transformer_options=transformer_options,
                **extra_kwargs,
            )

        if control is not None:
            control_o = control.get("output")
            if control_o is not None and block_index < len(control_o):
                add = control_o[block_index]
                if add is not None:
                    img[:, txt.shape[1] : txt.shape[1] + add.shape[1], ...] += add

    prehead_feature = img[:, txt.shape[1] :, ...]
    if run_id is not None and solver_step_id is not None:
        runtime.observe_actual_feature(run_id, solver_step_id, prehead_feature)

    final_kwargs = {}
    if modulation_dims is not None:
        final_kwargs["modulation_dims"] = modulation_dims
    return inner.final_layer(prehead_feature, vec_orig, **final_kwargs)


class FluxSpectrumPatcher:
    @staticmethod
    def patch(model: Any, cfg: SpectrumConfig) -> Any:
        cfg = cfg.validate()
        patched = _clone_model(model)
        transformer_options = _ensure_transformer_options(patched)
        runtime = SpectrumRuntime(cfg)
        transformer_options["spectrum_runtime"] = runtime
        transformer_options["spectrum_enabled"] = cfg.enabled
        transformer_options["spectrum_backend"] = "flux"
        transformer_options["spectrum_cfg"] = cfg.to_dict()
        _install_sampler_level_wrappers(patched, runtime)

        inner, path = locate_flux_inner_model(patched)
        if inner is None:
            raise RuntimeError(
                "Could not locate ComfyUI Flux diffusion model. This node currently supports native FLUX models only."
            )
        _install_generic_flux_wrapper(inner)
        if cfg.debug:
            LOG.warning("Spectrum installed on %s", path)
        return patched
