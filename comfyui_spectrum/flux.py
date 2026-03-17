from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .config import SpectrumConfig
from .runtime import SpectrumRuntime

LOG = logging.getLogger(__name__)


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

    decision = runtime.begin_step(transformer_options, timesteps)
    step_idx = decision["step_idx"]
    total_steps = decision["total_steps"]
    actual_forward = decision["actual_forward"]

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

    if not actual_forward and runtime.forecaster.ready(runtime.min_fit_points):
        pred_feature = runtime.forecaster.predict(
            step_index=step_idx,
            total_steps=total_steps,
            blend_weight=runtime.cfg.blend_weight,
        )
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
    runtime.forecaster.update(step_idx, prehead_feature)

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

        inner, path = locate_flux_inner_model(patched)
        if inner is None:
            raise RuntimeError(
                "Could not locate ComfyUI Flux diffusion model. This node currently supports native FLUX models only."
            )
        _install_generic_flux_wrapper(inner)
        if cfg.debug:
            LOG.warning("Spectrum installed on %s", path)
        return patched
