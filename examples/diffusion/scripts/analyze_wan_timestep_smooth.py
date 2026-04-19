#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze timestep-aware smooth statistics for Wan calibration caches.

This is an analysis-only script. It does not modify model weights or run the
full search-based smooth calibration. Instead, it:

1. Loads the Wan PTQ config and calibration caches.
2. Replays layer activation caching with the existing loader.
3. For each smoothable linear target, groups cached samples by timestep.
4. Computes:
   - `x_span[timestep, channel]`
   - `global_x_span[channel]`
   - `scale[timestep, channel]`
   - `global_scale[channel]`
5. Saves a compact JSON summary plus a `.pt` file with tensors.

By default, `scale` is computed for one chosen smooth candidate from the current
config's `(alpha, beta, span_pair)` candidate pool. This keeps output size and
runtime manageable for first-pass analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


REPO_ROOT = Path("/data1/lyf/Lab/VACE/deepcompressor_vace").resolve()
EXAMPLES_DIR = REPO_ROOT / "examples" / "diffusion"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import for side effects:
# - register Wan pipeline factories
# - override DiffusionCalibCacheLoaderConfig.build_loader -> WanCalibCacheLoader
import deepcompressor.app.diffusion.ptq_wan_t2v  # noqa: F401

from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.app.diffusion.nn.struct import DiffusionAttentionStruct, DiffusionFeedForwardStruct, DiffusionModelStruct
from deepcompressor.app.diffusion.quant.config import DiffusionQuantConfig
from deepcompressor.app.diffusion.quant.utils import get_needs_inputs_fn
from deepcompressor.calib.smooth import SmoothLinearCalibrator, get_smooth_scale, get_smooth_span
from deepcompressor.quantizer import Quantizer


@dataclass
class SmoothTarget:
    module_name: str
    module_key: str
    cache_key: str
    input_channels_dim: int
    weights: list[torch.Tensor]


def parse_args() -> argparse.Namespace:
    default_configs = [
        str(EXAMPLES_DIR / "configs" / "model" / "wan2.1-t2v-1.3b.yaml"),
        str(EXAMPLES_DIR / "configs" / "svdquant" / "nvfp4.yaml"),
    ]
    default_cache = str(
        EXAMPLES_DIR / "datasets" / "torch.bfloat16" / "ViDiT-Q-Wan2.1-1.3B" / "caches"
    )
    parser = argparse.ArgumentParser(description="Analyze timestep-aware smooth statistics for Wan.")
    parser.add_argument(
        "configs",
        nargs="*",
        default=default_configs,
        help="Config files loaded in order. Defaults to Wan 1.3B + nvfp4.",
    )
    parser.add_argument(
        "--cache-path",
        default=default_cache,
        help="Calibration cache path. Overrides quant.calib.path in the merged config.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXAMPLES_DIR / "runs" / "analysis" / "wan_timestep_smooth"),
        help="Directory to save analysis outputs.",
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=10,
        help="Smooth candidate index over config.smooth.proj candidate pool.",
    )
    parser.add_argument(
        "--module-regex",
        default=".*",
        help="Regex to filter module names for analysis.",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=-1,
        help="Only analyze the first N transformer blocks (-1 means all).",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=-1,
        help="Only analyze the first N matched smooth targets (-1 means all).",
    )
    return parser.parse_args()


def load_run_config(config_paths: list[str], cache_path: str) -> DiffusionPtqRunConfig:
    os.chdir(REPO_ROOT)
    parser = DiffusionPtqRunConfig.get_parser()
    config, _, _, _, _ = parser.parse_known_args(args=config_paths)
    assert isinstance(config, DiffusionPtqRunConfig)
    config.quant.calib.path = os.path.abspath(os.path.expanduser(cache_path))
    return config


def load_sample_timesteps(cache_dir: str) -> tuple[list[float], dict[float, list[int]], list[int], list[int]]:
    filepaths = sorted(str(p) for p in Path(cache_dir).glob("*.pt"))
    if not filepaths:
        raise FileNotFoundError(f"No .pt cache files found under {cache_dir}")
    timesteps: list[float] = []
    steps: list[int] = []
    guidances: list[int] = []
    groups: dict[float, list[int]] = defaultdict(list)
    for idx, fp in enumerate(filepaths):
        sample = torch.load(fp, map_location="cpu", weights_only=False)
        timestep = float(sample["input_kwargs"]["timestep"].reshape(-1)[0].item())
        timesteps.append(timestep)
        groups[timestep].append(idx)
        steps.append(int(sample.get("step", -1)))
        guidances.append(int(sample.get("guidance", -1)))
    return timesteps, dict(sorted(groups.items(), key=lambda kv: kv[0])), steps, guidances


def split_candidate(smooth_cfg, candidate_index: int) -> tuple[float, float, tuple]:
    alpha_beta_pairs = smooth_cfg.get_alpha_beta_pairs()
    span_pairs = smooth_cfg.spans
    population_size = len(alpha_beta_pairs) * len(span_pairs)
    if candidate_index < 0 or candidate_index >= population_size:
        raise ValueError(
            f"candidate_index={candidate_index} out of range, expected [0, {population_size - 1}]"
        )
    alpha_beta_id = candidate_index % len(alpha_beta_pairs)
    span_pair_id = candidate_index // len(alpha_beta_pairs)
    alpha, beta = alpha_beta_pairs[alpha_beta_id]
    span_pair = span_pairs[span_pair_id]
    return alpha, beta, span_pair


def tensor_cache_to_batched_tensor(tensor_cache) -> torch.Tensor:
    if not tensor_cache.data:
        raise ValueError("TensorCache has no data")
    if len(tensor_cache.data) == 1:
        return tensor_cache.data[0]
    return torch.cat(tensor_cache.data, dim=0)


def standardize_subset(tensor_cache, indices: list[int]) -> torch.Tensor:
    batched = tensor_cache_to_batched_tensor(tensor_cache)
    subset = batched[indices]
    channels_dim = tensor_cache.channels_dim
    return subset.view(-1, *subset.shape[channels_dim:])


def build_proj_calibrator(
    quant_config: DiffusionQuantConfig,
    module_key: str,
    *,
    channels_dim: int,
) -> SmoothLinearCalibrator:
    weight_quantizer = Quantizer(
        quant_config.wgts,
        key=module_key,
        low_rank=quant_config.wgts.low_rank,
    )
    ipt_cfg = quant_config.unsigned_ipts if module_key.lower().endswith("down_proj") else quant_config.ipts
    input_quantizer = Quantizer(
        ipt_cfg,
        channels_dim=channels_dim,
        key=module_key,
    )
    return SmoothLinearCalibrator(
        config=quant_config.smooth.proj,
        weight_quantizer=weight_quantizer,
        input_quantizer=input_quantizer,
        develop_dtype=quant_config.develop_dtype,
    )


def iter_wan_smooth_targets(layer, layer_cache: dict[str, object], quant_config: DiffusionQuantConfig) -> Iterable[SmoothTarget]:
    smooth_cfg = quant_config.smooth.proj
    for attn in layer.iter_attention_structs():
        assert isinstance(attn, DiffusionAttentionStruct)
        module_key = attn.qkv_proj_key
        needs_quant = (quant_config.enabled_wgts and quant_config.wgts.is_enabled_for(module_key)) or (
            quant_config.enabled_ipts and quant_config.ipts.is_enabled_for(module_key)
        )
        if needs_quant and smooth_cfg.is_enabled_for(module_key) and attn.q_proj_name in layer_cache:
            yield SmoothTarget(
                module_name=f"{attn.name}.qkv_proj",
                module_key=module_key,
                cache_key=attn.q_proj_name,
                input_channels_dim=-1,
                weights=[m.weight for m in attn.qkv_proj],
            )

        if not attn.is_self_attn() and attn.add_k_proj is not None:
            module_key = attn.add_qkv_proj_key
            needs_quant = (quant_config.enabled_wgts and quant_config.wgts.is_enabled_for(module_key)) or (
                quant_config.enabled_ipts and quant_config.ipts.is_enabled_for(module_key)
            )
            if needs_quant and smooth_cfg.is_enabled_for(module_key) and attn.add_k_proj_name in layer_cache:
                yield SmoothTarget(
                    module_name=f"{attn.name}.add_qkv_proj",
                    module_key=module_key,
                    cache_key=attn.add_k_proj_name,
                    input_channels_dim=-1,
                    weights=[m.weight for m in attn.add_qkv_proj],
                )

        module_key = attn.out_proj_key
        needs_quant = (quant_config.enabled_wgts and quant_config.wgts.is_enabled_for(module_key)) or (
            quant_config.enabled_ipts and quant_config.ipts.is_enabled_for(module_key)
        )
        if needs_quant and smooth_cfg.is_enabled_for(module_key) and attn.o_proj_name in layer_cache:
            yield SmoothTarget(
                module_name=f"{attn.name}.out_proj",
                module_key=module_key,
                cache_key=attn.o_proj_name,
                input_channels_dim=-1,
                weights=[attn.o_proj.weight],
            )

    for ffn in (layer.ffn_struct, layer.add_ffn_struct):
        if ffn is None:
            continue
        assert isinstance(ffn, DiffusionFeedForwardStruct)
        module_key = ffn.up_proj_key
        needs_quant = (quant_config.enabled_wgts and quant_config.wgts.is_enabled_for(module_key)) or (
            quant_config.enabled_ipts and quant_config.ipts.is_enabled_for(module_key)
        )
        if needs_quant and smooth_cfg.is_enabled_for(module_key) and ffn.up_proj_name in layer_cache:
            yield SmoothTarget(
                module_name=f"{ffn.name}.up_proj",
                module_key=module_key,
                cache_key=ffn.up_proj_name,
                input_channels_dim=-1,
                weights=[m.weight for m in ffn.up_projs],
            )

        module_key = ffn.down_proj_key.upper()
        needs_quant = (quant_config.enabled_wgts and quant_config.wgts.is_enabled_for(module_key)) or (
            quant_config.enabled_ipts and quant_config.ipts.is_enabled_for(module_key)
        )
        if needs_quant and smooth_cfg.is_enabled_for(module_key) and ffn.down_proj_name in layer_cache:
            yield SmoothTarget(
                module_name=f"{ffn.name}.down_proj",
                module_key=module_key,
                cache_key=ffn.down_proj_name,
                input_channels_dim=-1,
                weights=[ffn.down_proj.weight],
            )


def analyze_target(
    target: SmoothTarget,
    layer_cache: dict[str, object],
    quant_config: DiffusionQuantConfig,
    timestep_groups: dict[float, list[int]],
    alpha: float,
    beta: float,
    span_pair: tuple,
) -> dict[str, object]:
    cache = layer_cache[target.cache_key].inputs.front()
    if not cache.data:
        raise ValueError(f"No cached inputs for {target.cache_key}")

    calibrator = build_proj_calibrator(
        quant_config,
        target.module_key,
        channels_dim=target.input_channels_dim,
    )
    x_span_mode, w_span_mode = span_pair

    all_indices = list(range(tensor_cache_to_batched_tensor(cache).shape[0]))
    global_x_tensor = standardize_subset(cache, all_indices)
    global_x_span = get_smooth_span(
        [global_x_tensor],
        group_shape=calibrator.x_group_shape,
        span_mode=x_span_mode,
        dtype=quant_config.develop_dtype,
    ).cpu()
    global_w_span = get_smooth_span(
        [w.data for w in target.weights],
        group_shape=calibrator.w_group_shape,
        span_mode=w_span_mode,
        dtype=quant_config.develop_dtype,
    ).cpu()
    global_scale = get_smooth_scale(
        alpha_base=global_x_span.clone(),
        beta_base=global_w_span.clone(),
        alpha=alpha,
        beta=beta,
    ).cpu()

    timesteps = []
    counts = []
    x_spans = []
    scales = []
    rel_l1 = []
    rel_linf = []
    eps = 1e-12
    for timestep, indices in timestep_groups.items():
        x_tensor = standardize_subset(cache, indices)
        x_span = get_smooth_span(
            [x_tensor],
            group_shape=calibrator.x_group_shape,
            span_mode=x_span_mode,
            dtype=quant_config.develop_dtype,
        ).cpu()
        scale = get_smooth_scale(
            alpha_base=x_span.clone(),
            beta_base=global_w_span.clone(),
            alpha=alpha,
            beta=beta,
        ).cpu()
        delta = (scale - global_scale).abs()
        denom = global_scale.abs().clamp_min(eps)
        rel = delta / denom
        timesteps.append(timestep)
        counts.append(len(indices))
        x_spans.append(x_span)
        scales.append(scale)
        rel_l1.append(rel.mean())
        rel_linf.append(rel.max())

    x_span_by_timestep = torch.stack(x_spans, dim=0)
    scale_by_timestep = torch.stack(scales, dim=0)
    rel_l1_by_timestep = torch.stack(rel_l1).cpu()
    rel_linf_by_timestep = torch.stack(rel_linf).cpu()

    summary = {
        "module_name": target.module_name,
        "module_key": target.module_key,
        "cache_key": target.cache_key,
        "num_channels": int(global_scale.numel()),
        "num_timesteps": len(timesteps),
        "timestep_values": timesteps,
        "sample_counts_per_timestep": counts,
        "span_pair": [x_span_mode.name, w_span_mode.name],
        "alpha": alpha,
        "beta": beta,
        "mean_rel_l1": float(rel_l1_by_timestep.mean().item()),
        "max_rel_l1": float(rel_l1_by_timestep.max().item()),
        "mean_rel_linf": float(rel_linf_by_timestep.mean().item()),
        "max_rel_linf": float(rel_linf_by_timestep.max().item()),
    }
    details = {
        "summary": summary,
        "global_x_span": global_x_span,
        "global_w_span": global_w_span,
        "global_scale": global_scale,
        "x_span_by_timestep": x_span_by_timestep,
        "scale_by_timestep": scale_by_timestep,
        "rel_l1_by_timestep": rel_l1_by_timestep,
        "rel_linf_by_timestep": rel_linf_by_timestep,
        "timesteps": torch.tensor(timesteps, dtype=torch.float32),
        "sample_counts_per_timestep": torch.tensor(counts, dtype=torch.int64),
    }
    return details


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_run_config(args.configs, args.cache_path)
    quant_config = config.quant
    assert quant_config.enabled_smooth and quant_config.smooth.enabled_proj, "Projected smooth must be enabled"

    timesteps, timestep_groups, steps, guidances = load_sample_timesteps(args.cache_path)
    if len(set(Path(fp).stem.split("-")[0] for fp in map(str, Path(args.cache_path).glob("*.pt")))) != 1:
        print("[warn] Cache path contains multiple sample_id groups; analysis will still run in file order.")

    alpha, beta, span_pair = split_candidate(quant_config.smooth.proj, args.candidate_index)
    print(
        f"[info] candidate={args.candidate_index} "
        f"alpha={alpha:.4f} beta={beta:.4f} "
        f"span_pair=({span_pair[0].name}, {span_pair[1].name})"
    )
    print(
        f"[info] num_cache_files={len(timesteps)} "
        f"unique_steps={len(set(steps))} "
        f"unique_guidances={sorted(set(guidances))} "
        f"unique_timestep_values={len(timestep_groups)}"
    )

    pipeline = config.pipeline.build()
    model = DiffusionModelStruct.construct(pipeline)
    loader = quant_config.calib.build_loader()

    module_pattern = re.compile(args.module_regex)
    details: dict[str, dict[str, object]] = {}
    summaries: list[dict[str, object]] = []
    matched_targets = 0

    iterable = loader.iter_layer_activations(
        model,
        needs_inputs_fn=get_needs_inputs_fn(model, quant_config),
        skip_pre_modules=True,
        skip_post_modules=True,
    )
    for block_idx, (_, (layer, layer_cache, _layer_kwargs)) in enumerate(iterable):
        if args.max_blocks > 0 and block_idx >= args.max_blocks:
            break
        for target in iter_wan_smooth_targets(layer, layer_cache, quant_config):
            if not module_pattern.search(target.module_name):
                continue
            if args.max_targets > 0 and matched_targets >= args.max_targets:
                break
            matched_targets += 1
            print(f"[analyze] {target.module_name}")
            target_details = analyze_target(
                target=target,
                layer_cache=layer_cache,
                quant_config=quant_config,
                timestep_groups=timestep_groups,
                alpha=alpha,
                beta=beta,
                span_pair=span_pair,
            )
            details[target.module_name] = target_details
            summaries.append(target_details["summary"])
        if args.max_targets > 0 and matched_targets >= args.max_targets:
            break

    if not summaries:
        raise RuntimeError("No smooth targets matched the current filters.")

    meta = {
        "configs": args.configs,
        "cache_path": os.path.abspath(args.cache_path),
        "candidate_index": args.candidate_index,
        "alpha": alpha,
        "beta": beta,
        "span_pair": [span_pair[0].name, span_pair[1].name],
        "num_cache_files": len(timesteps),
        "unique_steps": len(set(steps)),
        "unique_guidances": sorted(set(guidances)),
        "unique_timestep_values": len(timestep_groups),
        "timestep_values": sorted(timestep_groups.keys()),
    }
    summary_payload = {"meta": meta, "modules": summaries}

    summary_path = output_dir / "summary.json"
    tensor_path = output_dir / "details.pt"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    torch.save({"meta": meta, "modules": details}, tensor_path)

    print(f"[done] wrote summary to {summary_path}")
    print(f"[done] wrote tensor details to {tensor_path}")


if __name__ == "__main__":
    main()
