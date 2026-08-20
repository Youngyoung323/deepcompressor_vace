# -*- coding: utf-8 -*-
"""Calibration data loader for Wan Video (VACE) models.

Loads the .pt calibration files collected by calib_wan.py and provides
them to the DeepCompressor quantization pipeline via BaseCalibCacheLoader.

Each .pt file has structure:
    {
        "input_args": [latent_tensor],          # (1, C, F, H, W)
        "input_kwargs": {
            "timestep": tensor,
            "context": tensor,
            "clip_feature": tensor_or_None,
            "y": tensor_or_None,
        },
        "outputs": [noise_pred_tensor],
        "sample_id": str,
        "step": int,
        "guidance": int,
    }
"""

import os
import random
import typing as tp
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.data
from einops import rearrange

from deepcompressor.data.cache import (
    IOTensorsCache,
    ModuleForwardInput,
    TensorCache,
    TensorsCache,
)
from deepcompressor.data.utils.reshape import LinearReshapeFn
from deepcompressor.dataset.action import CacheAction, ConcatCacheAction
from deepcompressor.dataset.cache import BaseCalibCacheLoader
from deepcompressor.utils.common import tree_copy_with_ref, tree_map

from ..nn.wan_struct import VaceWanDiTStruct, VaceWanTransformerBlockStruct, WanDiTStruct, WanTransformerBlockStruct

__all__ = ["WanCalibDataset", "WanCalibCacheLoader", "WanConcatCacheAction"]

try:
    from diffsynth.models.wan_video_dit import CrossAttention as _WanCrossAttention
    from diffsynth.models.wan_video_dit import SelfAttention as _WanSelfAttention
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d as _wan_sinusoidal_embedding_1d
except ImportError:  # optional; ptq_vace adds DiffSynth to sys.path

    class _WanAttentionPlaceholder(nn.Module):
        pass

    _WanSelfAttention = _WanCrossAttention = _WanAttentionPlaceholder
    _wan_sinusoidal_embedding_1d = None


class _VaceWanReplayModel(nn.Module):
    """Replay VACE side branch from model_fn-style calibration caches."""

    def __init__(self, vace: nn.Module, dit: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_dit", dit)
        self.vace_patch_embedding = vace.vace_patch_embedding
        self.vace_blocks = vace.vace_blocks

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        vace_context: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        vace_scale: float = 1.0,
        **kwargs,
    ):
        if vace_context is None:
            raise ValueError("VACE calibration cache is missing `vace_context`.")
        if _wan_sinusoidal_embedding_1d is None:
            raise ImportError("DiffSynth Wan utilities are required for VACE replay.")

        dit = self.__dict__["_dit"]
        t = dit.time_embedding(_wan_sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        context = dit.text_embedding(context)

        x = latents
        if x.shape[0] != context.shape[0]:
            x = torch.concat([x] * context.shape[0], dim=0)
        if y is not None and dit.require_vae_embedding:
            x = torch.cat([x, y], dim=1)
        if clip_feature is not None and dit.require_clip_embedding:
            clip_embedding = dit.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        patchified = dit.patchify(x)
        if isinstance(patchified, tuple):
            x, (f, h, w) = patchified
        else:
            x = patchified
            f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        freqs = torch.cat([
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))], dim=1)
            for u in c
        ])
        for block in self.vace_blocks:
            c = block(c, x, context, t_mod, freqs)
        return torch.unbind(c)[:-1]


class WanConcatCacheAction(ConcatCacheAction):
    """Like :class:`ConcatCacheAction`, but align Wan ``SelfAttention`` freqs layout with batch dim.

    WanModel passes ``freqs`` as ``(L, 1, D)`` (one slice per token) while ``x`` is ``(1, L, C)``.
    :class:`ConcatCacheAction` and :meth:`TensorCache.repartition` treat dim0 as batch/sample count,
    which breaks multi-arg caches and triggers ``AssertionError`` in calibration search.
    Prepend a batch dimension: ``(L, 1, D) -> (1, L, 1, D)``, which matches ``x``'s batch=1 and
    preserves broadcasting inside ``rope_apply`` (equivalent to ``(L, 1, D)`` vs ``(1, L, n, …)``).
    """

    @staticmethod
    def _align_wan_self_attn_freqs(
        module: nn.Module, tensors: dict[int | str, torch.Tensor]
    ) -> dict[int | str, torch.Tensor]:
        if not isinstance(module, _WanSelfAttention):
            return tensors
        if "x" not in tensors or "freqs" not in tensors:
            return tensors
        x, freqs = tensors["x"], tensors["freqs"]
        if not isinstance(x, torch.Tensor) or not isinstance(freqs, torch.Tensor):
            return tensors
        # x: (1, L, C); freqs from DiT: (f*h*w, 1, D) i.e. (L, 1, D)
        if (
            x.dim() == 3
            and freqs.dim() == 3
            and x.shape[0] == 1
            and freqs.shape[0] == x.shape[1]
            and freqs.shape[1] == 1
        ):
            out = dict(tensors)
            out["freqs"] = freqs.unsqueeze(0)
            return out
        return tensors

    def apply(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        tensors = self._align_wan_self_attn_freqs(module, tensors)
        return super().apply(name, module, tensors, cache)

    def info(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        tensors = self._align_wan_self_attn_freqs(module, tensors)
        return super().info(name, module, tensors, cache)


class WanCalibDataset(torch.utils.data.Dataset):
    """Dataset that loads pre-collected .pt calibration files."""

    data: list[dict[str, tp.Any]]

    def __init__(self, path: str, num_samples: int = -1, seed: int = 0) -> None:
        if os.path.isdir(path):
            filepaths = sorted(
                os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pt")
            )
        else:
            filepaths = [path]
        """
        data = [torch.load(fp, weights_only=False) for fp in filepaths]
        random.Random(seed).shuffle(data)
        if 0 < num_samples < len(data):
            data = data[:num_samples]
        self.data = data
        """
        if num_samples > 0 and num_samples < len(filepaths):
            random.Random(seed).shuffle(filepaths)
            filepaths = filepaths[:num_samples]
            filepaths = sorted(filepaths)
        self.data = [torch.load(fp, weights_only=False) for fp in filepaths]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, tp.Any]:
        return self.data[idx]


class WanCalibCacheLoader(BaseCalibCacheLoader):
    """Calibration cache loader for WanModel.

    Wraps WanCalibDataset and feeds samples into the
    DeepCompressor iter_layer_activations framework.

    Note:
        ``batch_size`` is forced to ``1``. Samples with a different latent shape than
        the first (after shuffle) are **dropped** — ``ConcatCacheAction`` stacks
        activations as ``(N, …)`` and requires identical token/spatial dims for all N.
        YAML ``calib.batch_size`` is ignored for Wan.
    """

    dataset: WanCalibDataset

    def __init__(
        self,
        path: str,
        num_samples: int = -1,
        batch_size: int = 1,
        seed: int = 0,
    ) -> None:
        dataset = WanCalibDataset(path, num_samples=num_samples, seed=seed)
        """
        if batch_size != 1:
            warnings.warn(
                "WanCalibCacheLoader: calibration samples may have different latent "
                f"shapes; ignoring batch_size={batch_size} and using batch_size=1.",
                UserWarning,
                stacklevel=2,
            )
            batch_size = 1
        """
        super().__init__(dataset=dataset, batch_size=min(batch_size, len(dataset)))

    """
    def _layer_forward_pre_hook(
        self,
        m: nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, tp.Any],
        cache: list[ModuleForwardInput],
        save_all: bool = False,
    ) -> None:
        """"""Same as ``BaseCalibCacheLoader``, but tolerate varying token lengths across samples.

        Default ``tree_copy_with_ref`` requires identical tensor shapes so cached references
        can be shared; Wan calibration clips have different ``F×H×W`` → different ``L=f*h*w``,
        so we fall back to independent tensors when shapes diverge.
        """"""
        inputs = self._convert_layer_inputs(m, args, kwargs, save_all=save_all)
        if len(cache) > 0:
            try:
                inputs.args = tree_copy_with_ref(inputs.args, cache[0].args)
                inputs.kwargs = tree_copy_with_ref(inputs.kwargs, cache[0].kwargs)
            except AssertionError:
                inputs.args = tree_map(lambda x: x, inputs.args)
                inputs.kwargs = tree_map(lambda x: x, inputs.kwargs)
        else:
            inputs.args = tree_map(lambda x: x, inputs.args)
            inputs.kwargs = tree_map(lambda x: x, inputs.kwargs)
        cache.append(inputs)
    """

    def iter_samples(self) -> tp.Generator[ModuleForwardInput, None, None]:
        for i in range(0, len(self.dataset), self.batch_size):
            batch = [self.dataset[j] for j in range(i, min(i + self.batch_size, len(self.dataset)))]
            args_list = [item["input_args"] for item in batch]
            kwargs_list = [item["input_kwargs"] for item in batch]

            if len(batch) == 1:
                yield ModuleForwardInput(args=args_list[0], kwargs=kwargs_list[0])
            else:
                batched_args = [
                    torch.cat([a[k] for a in args_list], dim=0) for k in range(len(args_list[0]))
                ]
                batched_kwargs = {}
                for key in kwargs_list[0]:
                    vals = [kw[key] for kw in kwargs_list]
                    if vals[0] is not None and isinstance(vals[0], torch.Tensor):
                        batched_kwargs[key] = torch.cat(vals, dim=0)
                    else:
                        batched_kwargs[key] = vals[0]
                yield ModuleForwardInput(args=batched_args, kwargs=batched_kwargs)

    def _init_cache(self, name: str, module: nn.Module) -> IOTensorsCache:
        if isinstance(module, nn.Linear):
            return IOTensorsCache(
                inputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                outputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
            )
        # Whole-module hooks on Wan SelfAttention / CrossAttention (see get_needs_inputs_fn:
        # q/k/v add parent attention module name). Keys must match forward() parameter names
        # for KeyedInputPackager (same idea as DiffusionCalibCacheLoader + diffusers Attention).
        # 与 DiffusionCalibCacheLoader 对 diffusers Attention 的 info 逻辑一致：3D 激活用
        # channels_dim=-1 + LinearReshapeFn，否则 smooth 里 eval_inputs.repartition 会因 dim=None 报错。
        if isinstance(module, _WanSelfAttention):
            return IOTensorsCache(
                inputs=TensorsCache(
                    OrderedDict(
                        x=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                        freqs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                    )
                ),
                outputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
            )
        if isinstance(module, _WanCrossAttention):
            return IOTensorsCache(
                inputs=TensorsCache(
                    OrderedDict(
                        x=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                        y=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                    )
                ),
                outputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
            )
        return super()._init_cache(name, module)

    def _convert_layer_inputs(
        self, m: nn.Module, args: tuple[tp.Any, ...], kwargs: dict[str, tp.Any], save_all: bool = False
    ) -> ModuleForwardInput:
        kwargs = {k: v for k, v in kwargs.items()}
        if "hidden_states" in kwargs:
            hidden_states = kwargs.pop("hidden_states")
            assert len(args) == 0
            remaining_args = []
        elif len(args) > 0:
            hidden_states = args[0]
            remaining_args = list(args[1:])
        else:
            hidden_states = None
            remaining_args = []

        from dataclasses import MISSING
        return ModuleForwardInput(
            args=[hidden_states.detach().cpu() if save_all and hidden_states is not None else MISSING, *remaining_args],
            kwargs=kwargs,
        )

    def _convert_layer_outputs(self, m: nn.Module, outputs: tp.Any) -> dict[str | int, tp.Any]:
        if isinstance(outputs, torch.Tensor):
            return {0: outputs.detach().cpu()}
        elif isinstance(outputs, (tuple, list)) and len(outputs) >= 1:
            return {0: outputs[0].detach().cpu() if isinstance(outputs[0], torch.Tensor) else outputs[0]}
        return super()._convert_layer_outputs(m, outputs)

    def iter_layer_activations(
        self,
        model: nn.Module | WanDiTStruct | VaceWanDiTStruct,
        *args,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool],
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | None = None,
        action: CacheAction | None = None,
        skip_pre_modules: bool = True,
        skip_post_modules: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[str, tuple[tp.Any, dict[str, IOTensorsCache], dict[str, tp.Any]]],
        None,
        None,
    ]:
        if isinstance(model, VaceWanDiTStruct):
            model_struct = model
            if model_struct.dit is None:
                raise ValueError("VaceWanDiTStruct requires `dit` for calibration replay.")
            model = _VaceWanReplayModel(model_struct.module, model_struct.dit)
        elif not isinstance(model, WanDiTStruct):
            model_struct = WanDiTStruct.construct(model)
        else:
            model_struct = model
            model = model_struct.module
        assert isinstance(model_struct, (WanDiTStruct, VaceWanDiTStruct))
        assert isinstance(model, nn.Module)

        action = WanConcatCacheAction("cpu") if action is None else action

        layers, layer_structs, recomputes, use_prev_layer_outputs = model_struct.get_iter_layer_activations_args(
            skip_pre_modules=skip_pre_modules,
            skip_post_modules=skip_post_modules,
        )

        for layer_idx, (layer_name, (layer, layer_cache, layer_inputs)) in enumerate(
            self._iter_layer_activations(
                model,
                *args,
                action=action,
                layers=layers,
                needs_inputs_fn=needs_inputs_fn,
                needs_outputs_fn=needs_outputs_fn,
                recomputes=recomputes,
                use_prev_layer_outputs=use_prev_layer_outputs,
                **kwargs,
            )
        ):
            layer_kwargs = {k: v for k, v in layer_inputs[0].kwargs.items()}
            layer_kwargs.pop("hidden_states", None)
            layer_struct = layer_structs[layer_idx]

            if isinstance(layer_struct, (WanTransformerBlockStruct, VaceWanTransformerBlockStruct)):
                assert layer_struct.name == layer_name
                assert layer is layer_struct.module
                for attn_struct in layer_struct.iter_attention_structs():
                    if attn_struct.q_proj_name in layer_cache:
                        if not attn_struct.is_cross_attn():
                            cache = layer_cache[attn_struct.q_proj_name]
                            layer_cache[attn_struct.k_proj_name] = cache
                            layer_cache[attn_struct.v_proj_name] = cache

            yield layer_name, (layer_struct, layer_cache, layer_kwargs)
