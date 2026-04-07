# -*- coding: utf-8 -*-
"""统计 Wan2.1-T2V-1.3B 校准目录下 .pt 条数，以及 YAML 下 WanCalibDataset 实际会用多少条。

运行（在仓库根目录）:
  python test.py

逻辑与 ``WanCalibDataset`` / ``quant.calib.path`` 的 format 规则对齐；不导入 ``DiffusionPtqRunConfig``（避免依赖 diffusers）。
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
EXAMPLES_DIFFUSION = REPO_ROOT / "examples" / "diffusion"
DEFAULT_YAML = EXAMPLES_DIFFUSION / "configs" / "__default__.yaml"
WAN_13B_YAML = EXAMPLES_DIFFUSION / "configs" / "model" / "wan2.1-t2v-1.3b.yaml"

# 一次性 torch.load 全部 .pt 的安全上限（与 WanCalibDataset.__init__ 行为一致：会先全读再截断）


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_wan_13b_merged_config() -> dict:
    with open(DEFAULT_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(WAN_13B_YAML, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    return _deep_merge(cfg, model_cfg)


def _resolve_calib_path(cfg: dict) -> Path:
    pipeline = cfg["pipeline"]
    eval_cfg = cfg["eval"]
    quant_calib = cfg["quant"]["calib"]
    protocol_tmpl = eval_cfg["protocol"].lower()
    protocol = protocol_tmpl.format(
        num_steps=eval_cfg["num_steps"],
        guidance_scale=eval_cfg["guidance_scale"],
    )
    path_tmpl = quant_calib["path"]
    dtype = pipeline["dtype"]
    name = pipeline["name"]
    family = name.split("-")[0]
    resolved = path_tmpl.format(
        dtype=dtype,
        family=family,
        model=name,
        protocol=protocol,
        data=quant_calib["data"],
    )
    return (EXAMPLES_DIFFUSION / resolved).resolve()


def main() -> None:
    cfg = _load_wan_13b_merged_config()
    calib_path = _resolve_calib_path(cfg)
    num_samples_yaml = int(cfg["quant"]["calib"]["num_samples"])
    seed = int(cfg.get("seed", 12345))

    if not calib_path.is_dir():
        print(f"[错误] 校准目录不存在: {calib_path}")
        print("请确认已按 collect 脚本生成缓存，或检查 model yaml 中 quant.calib.path。")
        sys.exit(1)

    pt_files = sorted(glob.glob(str(calib_path / "*.pt")))
    n_disk = len(pt_files)

    # 与 WanCalibDataset: shuffle 后若 0 < num_samples < len(data) 则 data = data[:num_samples]
    if 0 < num_samples_yaml < n_disk:
        n_effective = num_samples_yaml
    else:
        n_effective = n_disk

    print("=== Wan2.1-T2V-1.3B 校准数据 .pt 统计 ===")
    print(f"模型配置: {WAN_13B_YAML}")
    print(f"解析后校准目录: {calib_path}")
    print(f"磁盘上 *.pt 文件数: {n_disk}")
    print(f"YAML quant.calib.num_samples: {num_samples_yaml}")
    print(f"与 WanCalibDataset 截断规则一致时，加载后条目数 len(dataset): {n_effective}")

    if n_disk == 0:
        return

    sys.path.insert(0, str(REPO_ROOT))
    from deepcompressor.app.diffusion.dataset.calib_wan_loader import WanCalibDataset

    ds = WanCalibDataset(str(calib_path), num_samples=num_samples_yaml, seed=seed)
    print(f"\n实例化 WanCalibDataset(seed={seed}, num_samples={num_samples_yaml}) -> len(dataset) = {len(ds)}")
    if len(ds) != n_effective:
        print(f"[警告] 与预期 {n_effective} 不一致，请检查实现是否有变。")


if __name__ == "__main__":
    os.chdir(EXAMPLES_DIFFUSION)
    main()
