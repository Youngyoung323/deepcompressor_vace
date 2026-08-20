# -*- coding: utf-8 -*-
"""Common utilities for collecting data."""

import inspect
import typing as tp

import torch
import torch.nn as nn
from diffusers.models.transformers import (
    FluxTransformer2DModel,
    PixArtTransformer2DModel,
    SanaTransformer2DModel,
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

try:
    from diffsynth.models.wan_video_dit import WanModel
except ImportError:
    WanModel = None

from deepcompressor.utils.common import tree_map, tree_split

__all__ = ["CollectHook", "ModelFnCollectHook"]


class CollectHook:
    """Forward hook that captures model inputs/outputs for calibration.

    Registered via ``model.register_forward_hook(hook, with_kwargs=True)``.
    Supports UNet2DConditionModel, PixArt, Sana, Flux, and WanModel.

    Note: In WanVideo VACE pipelines, ``model_fn_wan_video`` accesses
    ``dit``'s submodules directly without calling ``dit.forward()``, so a
    forward_hook on ``dit`` will NOT fire. Use :class:`ModelFnCollectHook`
    to wrap ``pipeline.model_fn`` instead.
    """

    def __init__(self, caches: list[dict[str, tp.Any]] = None, zero_redundancy: bool = False) -> None:
        self.caches = [] if caches is None else caches
        self.zero_redundancy = zero_redundancy

    def __call__(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tuple[torch.Tensor, ...],
    ) -> tp.Any:
        new_args = []
        signature = inspect.signature(module.forward)
        bound_arguments = signature.bind(*input_args, **input_kwargs)
        arguments = bound_arguments.arguments
        args_to_kwargs = {k: v for k, v in arguments.items() if k not in input_kwargs}
        input_kwargs.update(args_to_kwargs)

        if isinstance(module, UNet2DConditionModel):
            sample = input_kwargs.pop("sample")
            new_args.append(sample)
            timestep = input_kwargs["timestep"]
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])
            input_kwargs["timestep"] = timesteps
        elif isinstance(module, (PixArtTransformer2DModel, SanaTransformer2DModel)):
            new_args.append(input_kwargs.pop("hidden_states"))
        elif isinstance(module, FluxTransformer2DModel):
            new_args.append(input_kwargs.pop("hidden_states"))
        #elif WanModel is not None and isinstance(module, WanModel):
        #    new_args.append(input_kwargs.pop("x"))
        #    input_kwargs.pop("use_gradient_checkpointing", None)
        #    input_kwargs.pop("use_gradient_checkpointing_offload", None)
        #    for k in list(input_kwargs):
        #        if not isinstance(input_kwargs[k], torch.Tensor):
        #            input_kwargs.pop(k)
        else:
            raise ValueError(f"Unknown model: {module}")
        cache = tree_map(lambda x: x.cpu(), {"input_args": new_args, "input_kwargs": input_kwargs, "outputs": output})
        split_cache = tree_split(cache)

        if isinstance(module, PixArtTransformer2DModel) and self.zero_redundancy:
            for cache in split_cache:
                cache_kwargs = cache["input_kwargs"]
                encoder_hidden_states = cache_kwargs.pop("encoder_hidden_states")
                assert encoder_hidden_states.shape[0] == 1
                encoder_attention_mask = cache_kwargs.get("encoder_attention_mask", None)
                if encoder_attention_mask is not None:
                    encoder_hidden_states = encoder_hidden_states[:, : max(encoder_attention_mask.sum(), 1)]
                cache_kwargs["encoder_hidden_states"] = encoder_hidden_states

        self.caches.extend(split_cache)


class ModelFnCollectHook:
    """Wraps a pipeline's ``model_fn`` to collect calibration data.
    Usage in ``calib_wan.py``::

        hook = ModelFnCollectHook(model_fn=pipeline.model_fn, caches=caches)
        pipeline.model_fn = hook
    """

    KWARGS_TENSOR_KEYS = ("timestep", "context", "clip_feature", "y", "vace_context")
    KWARGS_VALUE_KEYS = ("vace_scale",)

    def __init__(
        self,
        model_fn: tp.Callable,
        caches: list[dict[str, tp.Any]] | None = None,
    ) -> None:
        self.model_fn = model_fn
        self.caches = [] if caches is None else caches

    def __call__(self, **kwargs) -> tp.Any:
        output = self.model_fn(**kwargs)

        input_args = []
        latents = kwargs.get("latents")
        if latents is not None and isinstance(latents, torch.Tensor):
            input_args.append(latents)

        input_kwargs: dict[str, tp.Any] = {}
        for key in self.KWARGS_TENSOR_KEYS:
            val = kwargs.get(key)
            if val is not None and isinstance(val, torch.Tensor):
                input_kwargs[key] = val
        for key in self.KWARGS_VALUE_KEYS:
            val = kwargs.get(key)
            if val is not None:
                input_kwargs[key] = val

        cache = tree_map(
            lambda x: x.cpu(),
            {"input_args": input_args, "input_kwargs": input_kwargs, "outputs": output},
        )
        split_cache = tree_split(cache)
        self.caches.extend(split_cache)

        return output
