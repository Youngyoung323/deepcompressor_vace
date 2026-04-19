#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze timestep-wise activation maxima from `details.pt`.

This script reads the output of `analyze_wan_timestep_smooth.py` and keeps only
the simplest statistic:

- for each module
- for each timestep
- the maximum activation magnitude

Given `x_span_by_timestep[T, C]`, where each row already stores channel-wise
activation maxima for one timestep, this script reduces over channels and saves:

- `per_timestep_activation_max[t] = max_c x_span_by_timestep[t, c]`

It also optionally saves a simple line plot.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Wan timestep smooth details.pt outputs.")
    parser.add_argument(
        "details_pt",
        nargs="?",
        default="/data1/lyf/Lab/VACE/deepcompressor_vace/examples/diffusion/runs/analysis/wan_timestep_smooth_smoke/details.pt",
        help="Path to details.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to save analysis outputs. Defaults to sibling folder of details.pt.",
    )
    parser.add_argument(
        "--module-regex",
        default=".*",
        help="Regex to filter module names.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k timesteps to report per module.",
    )
    parser.add_argument(
        "--max-modules",
        type=int,
        default=-1,
        help="Analyze at most N modules (-1 means all).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only save JSON summaries, skip PNG plots.",
    )
    return parser.parse_args()


def tensor_topk_dict(values: torch.Tensor, labels: list[float | int], k: int) -> list[dict[str, float | int]]:
    k = min(k, values.numel())
    topk = torch.topk(values, k=k)
    result = []
    for idx, val in zip(topk.indices.tolist(), topk.values.tolist(), strict=True):
        result.append({"index": int(idx), "label": labels[idx], "value": float(val)})
    return result


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def make_module_summary(name: str, module_data: dict[str, object], topk: int) -> dict[str, object]:
    timesteps_tensor = module_data["timesteps"].cpu()
    timestep_labels = [float(x) for x in timesteps_tensor.tolist()]

    x_span_by_timestep = module_data["x_span_by_timestep"].cpu()
    global_x_span = module_data["global_x_span"].cpu()
    per_timestep_activation_max = x_span_by_timestep.max(dim=1).values
    global_activation_max = global_x_span.max()

    return {
        "module_name": name,
        "input_summary": module_data["summary"],
        "global_activation_max": float(global_activation_max.item()),
        "per_timestep_activation_max": [
            {"index": int(i), "label": timestep_labels[i], "value": float(v)}
            for i, v in enumerate(per_timestep_activation_max.tolist())
        ],
        "top_timesteps_by_activation_max": tensor_topk_dict(
            per_timestep_activation_max, timestep_labels, topk
        ),
    }


def save_plots(
    name: str,
    module_data: dict[str, object],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    timesteps = module_data["timesteps"].cpu().numpy()
    x_span_by_timestep = module_data["x_span_by_timestep"].cpu()
    per_timestep_activation_max = x_span_by_timestep.max(dim=1).values.numpy()

    stem = sanitize_filename(name)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, per_timestep_activation_max, marker="o", markersize=3)
    ax.set_title(f"{name} | per-timestep activation max")
    ax.set_xlabel("timestep")
    ax.set_ylabel("activation max")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.activation_max_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    details_path = Path(args.details_pt).resolve()
    if not details_path.exists():
        raise FileNotFoundError(details_path)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = details_path.parent / f"{details_path.stem}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(details_path, map_location="cpu", weights_only=False)
    module_pattern = re.compile(args.module_regex)

    all_modules: dict[str, dict[str, object]] = data["modules"]
    matched_names = [name for name in all_modules if module_pattern.search(name)]
    if args.max_modules > 0:
        matched_names = matched_names[: args.max_modules]

    if not matched_names:
        raise RuntimeError("No modules matched the given regex/filter.")

    summaries: list[dict[str, object]] = []
    for name in matched_names:
        module_data = all_modules[name]
        summary = make_module_summary(name, module_data, args.topk)
        summaries.append(summary)
        print(f"[analyze] {name}")
        print(
            f"  global_activation_max={summary['global_activation_max']:.6f}"
        )
        if not args.no_plots:
            save_plots(name, module_data, output_dir)

    payload = {
        "meta": data["meta"],
        "details_path": str(details_path),
        "num_modules_analyzed": len(matched_names),
        "modules": summaries,
    }
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[done] wrote summary to {summary_path}")
    if not args.no_plots:
        print(f"[done] wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
