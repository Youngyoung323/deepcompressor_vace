# -*- coding: utf-8 -*-
"""分析单个 Wan T2V 校准 .pt 文件的结构（键、嵌套、张量 shape/dtype）。

默认文件:
  examples/diffusion/datasets/torch.bfloat16/Wan2.1-T2V-1.3B-50steps/Ditto-1M/s128/caches/000001-00000-0.pt

用法:
  python test_calib.py
  python test_calib.py /path/to/other.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

DEFAULT_PT = (
    "/data1/lyf/Lab/VACE/deepcompressor_vace/examples/diffusion/datasets/torch.bfloat16/Wan2.1-VACE-14B-lora-20steps/VACE-benchmark-real/s1/caches/000010-00000-0.pt"
)


def _tensor_brief(t: torch.Tensor, indent: str) -> None:
    dev = t.device
    print(f"{indent}  shape: {tuple(t.shape)}")
    print(f"{indent}  dtype: {t.dtype}")
    print(f"{indent}  device: {dev}")
    print(f"{indent}  numel: {t.numel()}  requires_grad: {t.requires_grad}")
    if t.numel() <= 16 and t.numel() > 0:
        print(f"{indent}  values: {t.detach().cpu().flatten().tolist()}")
    elif t.numel() > 0:
        t_flat = t.detach().float().flatten()
        print(
            f"{indent}  stats (float view): min={t_flat.min().item():.6g} "
            f"max={t_flat.max().item():.6g} mean={t_flat.mean().item():.6g}"
        )


def describe(obj: Any, name: str = "root", indent: str = "", max_depth: int = 12) -> None:
    if max_depth < 0:
        print(f"{indent}{name}: <max_depth reached>")
        return

    prefix = f"{indent}{name}: "

    if obj is None:
        print(f"{prefix}None")
        return

    if isinstance(obj, torch.Tensor):
        print(f"{prefix}Tensor")
        _tensor_brief(obj, indent)
        return

    if isinstance(obj, (bool, int, float, str, bytes)):
        print(f"{prefix}{type(obj).__name__} = {obj!r}")
        return

    if isinstance(obj, dict):
        print(f"{prefix}dict (len={len(obj)})")
        for k in sorted(obj.keys(), key=lambda x: (str(type(x)), str(x))):
            v = obj[k]
            describe(v, f"['{k}']" if isinstance(k, str) else f"[{k!r}]", indent + "  ", max_depth - 1)
        return

    if isinstance(obj, (list, tuple)):
        tag = "list" if isinstance(obj, list) else "tuple"
        print(f"{prefix}{tag} (len={len(obj)})")
        for i, item in enumerate(obj):
            describe(item, f"[{i}]", indent + "  ", max_depth - 1)
        return

    print(f"{prefix}{type(obj).__name__} (repr: {repr(obj)[:200]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 Wan 校准 .pt 结构")
    parser.add_argument(
        "pt_path",
        nargs="?",
        default=DEFAULT_PT,
        help="待加载的 .pt 路径（默认: 仓库内示例 caches 下指定文件）",
    )
    args = parser.parse_args()
    path = os.path.abspath(os.path.expanduser(args.pt_path))

    if not os.path.isfile(path):
        print(f"[错误] 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    print("=== Wan 校准 .pt 结构分析 ===")
    print(f"路径: {path}")
    print(f"文件大小: {os.path.getsize(path) / (1024 * 1024):.4f} MiB")
    print()

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[错误] torch.load 失败: {e}", file=sys.stderr)
        sys.exit(1)

    describe(payload, "payload")


if __name__ == "__main__":
    main()
