#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析不同 timestep 下同一通道的激活值分布。

直接加载校准集 .pt 文件 + 模型，用 forward hook 采集指定层的激活值，
按 timestep 分组统计每通道的 abs_max / mean / std，输出 .pt 结果 + 可视化。

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/analyze_activation_by_timestep.py \
        --calib-dir examples/diffusion/datasets/torch.bfloat16/Wan2.1-T2V-1.3B-50steps/Ditto-1M/s128/caches \
        --model-path /data1/lyf/Lab/DiffSynth-Studio/models \
        --output-dir analysis_output \
        --num-samples 1 \
        --target-layers "blocks.0.self_attn.q,blocks.0.ffn.0"

--num-samples: 使用几个 sample_id（每个有 50 step × 2 guidance = 100 个 .pt）
--target-layers: 逗号分隔的模块名子串（匹配 named_modules 中的 name）
                 留空则自动选取第 0 / 中间 / 最后一个 block 的 q_proj + up_proj
--fixed-channels: 固定 12 个通道索引（逗号分隔），用于 curves / channel_diff / channel_step_diff 等图，
                  便于有/无 rotation 时使用相同通道对比
--plots: 可选 channel_step_diff，红线表示相对前一个 denoising timestep 的逐步差分
--compare-rotation: 一次运行中分别采集无/有 rotation 的激活，统计全通道放大/缩小数量与幅度，
                    并分析同一通道跨 timestep 波动在 rotation 前后的变化
--compare-with: 与已有 baseline .pt 结果对比（当前 run 作为 rotation 侧）
--plots activation_3d: 仅生成 3D 图（X=Channel, Y=Token, Z=|activation|），需配合
                  --compare-rotation 或 --compare-with + --rotation；不会生成 heatmap 等
--activation-3d-timesteps: 3D 图使用的 timestep，逗号分隔，如 "0" 或 "999,500,0"
--activation-3d-max-channels / --activation-3d-max-tokens: 3D 图下采样上限
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/data1/lyf/Lab/DiffSynth-Studio")
##sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

# ── Model loading (reuse ptq_wan_t2v pipeline factory) ──────────────────
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


def build_pipeline(model_path: str, dtype=torch.bfloat16, device="cuda"):
    model_configs = [
        ModelConfig(path=os.path.join(
            model_path,
            "Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        )),
        ModelConfig(path=os.path.join(
            model_path,
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
            "models_t5_umt5-xxl-enc-bf16.safetensors",
        )),
        ModelConfig(path=os.path.join(
            model_path,
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
            "Wan2.1_VAE.safetensors",
        )),
    ]
    tokenizer_config = ModelConfig(
        path=os.path.join(model_path, "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
    )
    pipeline = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    return pipeline


# ── Calibration file loading ────────────────────────────────────────────
def load_calib_files(calib_dir: str, num_samples: int = 1):
    """返回 list of dict, 每个 dict 是一个校准 .pt 文件的内容。
    按 (sample_id, step, guidance) 排序。"""
    all_files = sorted(glob.glob(os.path.join(calib_dir, "*.pt")))
    if not all_files:
        raise FileNotFoundError(f"No .pt files found in {calib_dir}")

    sample_ids = sorted(set(os.path.basename(f).split("-")[0] for f in all_files))
    selected = sample_ids[:num_samples]
    print(f"Total {len(all_files)} files, {len(sample_ids)} samples. "
          f"Using {len(selected)} sample(s): {selected}")

    samples = []
    for f in all_files:
        sid = os.path.basename(f).split("-")[0]
        if sid in selected:
            samples.append(torch.load(f, weights_only=False))
    return samples


# ── Hadamard rotation ──────────────────────────────────────────────────
def _get_rotation_matrix(dim: int, device: str = "cuda"):
    """Get a Hadamard-based orthogonal rotation for the given dimension.
    Returns (mode, ...) tuple consumed by _apply_rotation()."""
    try:
        from deepcompressor.utils.math.hadamard import HadamardMatrix
        rhs, lhs, k = HadamardMatrix.get(dim, scale=True, dtype=torch.float32, device=device)
        return ("had", rhs, lhs, k)
    except Exception:
        print(f"  [rotation] DeepCompressor Hadamard unavailable for dim={dim}, "
              "falling back to random orthogonal (QR)")
        M = torch.randn(dim, dim, dtype=torch.float64)
        Q, R = torch.linalg.qr(M)
        Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
        return ("qr", Q.to(device=device, dtype=torch.float32))


def _apply_rotation(x: torch.Tensor, rot_info: tuple) -> torch.Tensor:
    """Apply the orthogonal rotation to the last dimension of x."""
    orig_dtype = x.dtype
    if rot_info[0] == "had":
        from deepcompressor.utils.math.hadamard import hardmard_transform
        _, rhs, lhs, k = rot_info
        return hardmard_transform(
            x.float(), rhs.to(x.device), lhs.to(x.device), k, scaled=True
        ).to(orig_dtype)
    else:
        _, Q = rot_info
        return (x.float() @ Q.to(x.device)).to(orig_dtype)


@torch.inference_mode()
def apply_hadamard_rotation(dit: nn.Module, device: str = "cuda") -> list:
    """Apply per-layer Hadamard rotation to all Linear layers in DiT blocks.

    For each Linear layer:
      - Offline: W_new = W @ Q  (fuse rotation into weight)
      - Online:  register forward_pre_hook  x -> x @ Q

    Since Q is orthogonal, (x @ Q) @ (W @ Q)^T = x @ Q @ Q^T @ W^T = x @ W^T,
    so the model output is mathematically unchanged, but the ActivationCollector
    will capture the *rotated* input, which should show more uniform per-channel
    distributions (outliers spread across all channels).
    """
    rot_cache: dict[int, tuple] = {}
    hooks: list = []
    count = 0

    for name, mod in dit.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not any(p in name for p in ("self_attn.", "cross_attn.", "ffn.")):
            continue

        dim = mod.in_features
        if dim not in rot_cache:
            rot_cache[dim] = _get_rotation_matrix(dim, device)

        ri = rot_cache[dim]

        # offline: W → W @ Q
        mod.weight.data = _apply_rotation(mod.weight.data, ri)

        # online: x → x @ Q (pre-hook so ActivationCollector sees rotated input)
        def _make_hook(_ri):
            def fn(module, args):
                x = args[0]
                return (_apply_rotation(x, _ri),) + args[1:]
            return fn

        h = mod.register_forward_pre_hook(_make_hook(ri))
        hooks.append(h)
        count += 1

    print(f"  [rotation] Hadamard rotation applied to {count} Linear layers "
          f"(dims: {sorted(rot_cache.keys())})")
    return hooks


# ── Hook-based activation collector ────────────────────────────────────
class ActivationCollector:
    def __init__(self, model: nn.Module, target_substrings: list[str]):
        self.hooks = []
        self.data: dict[str, torch.Tensor] = {}
        self._register(model, target_substrings)

    def _register(self, model: nn.Module, target_substrings: list[str]):
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not any(sub in name for sub in target_substrings):
                continue
            handle = mod.register_forward_hook(self._make_hook(name))
            self.hooks.append(handle)
            print(f"  [hook] {name}")

    def _make_hook(self, name: str):
        collector = self

        def hook_fn(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
            else:
                x = input
            if isinstance(x, torch.Tensor):
                collector.data[name] = x.detach().float().cpu()

        return hook_fn

    def clear(self):
        self.data.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── Per-channel statistics ─────────────────────────────────────────────
def compute_channel_stats(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """x: (batch, tokens, channels) or (batch, channels). Returns per-channel stats."""
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim > 3:
        x = x.reshape(-1, x.shape[-1])
    return {
        "abs_max": x.abs().amax(dim=0),
        "mean": x.mean(dim=0),
        "std": x.std(dim=0),
        "abs_mean": x.abs().mean(dim=0),
    }


# ── Visualization ──────────────────────────────────────────────────────
AVAILABLE_PLOTS = (
    "heatmap", "curves", "summary", "channel_diff", "channel_step_diff",
    "rotation_compare", "activation_3d",
)


def parse_fixed_channels(fixed_channels_str: str, n_channels: int) -> list[int]:
    """Parse comma-separated channel indices and validate against n_channels."""
    if not fixed_channels_str.strip():
        return []
    channels = [int(x.strip()) for x in fixed_channels_str.split(",") if x.strip()]
    if not channels:
        raise ValueError("--fixed-channels is set but no valid channel indices were parsed")
    invalid = [ch for ch in channels if ch < 0 or ch >= n_channels]
    if invalid:
        raise ValueError(
            f"Invalid channel indices {invalid} for n_channels={n_channels}"
        )
    return channels


def select_plot_channels(
    ts_data: dict,
    timesteps: list,
    n_channels: int,
    fixed_channels: list[int] | None,
    max_channels: int = 12,
) -> list[int]:
    """Return channel indices for curves / channel_diff plots."""
    if fixed_channels:
        return fixed_channels[:max_channels]

    channel_variance = torch.stack(
        [ts_data[t]["abs_max"] for t in timesteps]
    ).var(dim=0)
    top_channels = channel_variance.topk(min(8, n_channels)).indices.tolist()
    random_channels = list(range(0, n_channels, max(1, n_channels // 8)))[:8]
    return sorted(set(top_channels + random_channels))[:max_channels]


def _subplot_grid_shape(n_panels: int) -> tuple[int, int]:
    ncols = min(4, max(1, n_panels))
    nrows = (n_panels + ncols - 1) // ncols
    return nrows, ncols


def _compute_prev_step_delta(absmax_mat):
    """Delta vs previous denoising step: abs_max[t_prev] - abs_max[t].

    timesteps are sorted ascending; denoising order is high -> low, so t_prev
    is the next index in the ascending array.
    """
    import numpy as np

    step_delta = np.full_like(absmax_mat, np.nan)
    if absmax_mat.shape[0] > 1:
        step_delta[:-1, :] = absmax_mat[1:, :] - absmax_mat[:-1, :]
    return step_delta


def _plot_channel_absmax_grid(
    layer_name: str,
    timesteps: list,
    show_channels: list[int],
    absmax_mat,
    delta_mat,
    delta_label: str,
    stat_labels: tuple[str, str],
    suptitle_suffix: str,
    output_path: str,
    fixed_channels: list[int] | None,
    delta_markers: bool = False,
):
    import matplotlib.pyplot as plt
    import numpy as np

    n_panels = len(show_channels)
    nrows, ncols = _subplot_grid_shape(n_panels)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False
    )

    stat_primary = []
    stat_secondary = []
    if stat_labels[0] == "span":
        stat_primary = absmax_mat.max(axis=0) - absmax_mat.min(axis=0)
        stat_secondary = stat_primary / np.maximum(absmax_mat.max(axis=0), 1e-8)
    elif stat_labels[0] == "max_step":
        stat_primary = np.nanmax(np.abs(delta_mat), axis=0)
        stat_secondary = stat_primary / np.maximum(absmax_mat.max(axis=0), 1e-8)

    for idx, ch in enumerate(show_channels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        values = absmax_mat[:, idx]
        deltas = delta_mat[:, idx]
        ax.plot(timesteps, values, color="#1f77b4", linewidth=1.5, label="abs_max")
        delta_kwargs = dict(
            color="#d62728",
            linewidth=1.2,
            alpha=0.85,
            label=delta_label,
        )
        if delta_markers:
            delta_kwargs.update(marker="o", markersize=3.5, linestyle="-")
        else:
            delta_kwargs.update(linestyle="--")
        ax.plot(timesteps, deltas, **delta_kwargs)
        ax.axhline(values.max(), color="#1f77b4", linestyle=":", alpha=0.35)
        ax.axhline(values.min(), color="#1f77b4", linestyle=":", alpha=0.35)
        if stat_labels[1] == "rel":
            ax.set_title(
                f"ch {ch}\n{stat_labels[0]}={stat_primary[idx]:.4g}, "
                f"rel={stat_secondary[idx]:.2%}",
                fontsize=9,
            )
        else:
            ax.set_title(
                f"ch {ch}\n{stat_labels[0]}={stat_primary[idx]:.4g}, "
                f"{stat_labels[1]}={stat_secondary[idx]:.2%}",
                fontsize=9,
            )
        ax.set_xlabel("Timestep", fontsize=8)
        ax.set_ylabel("value", fontsize=8)
        ax.invert_xaxis()
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    fixed_note = (
        f"fixed channels: {show_channels}"
        if fixed_channels
        else f"auto-selected channels: {show_channels}"
    )
    fig.suptitle(
        f"{layer_name}\nPer-channel abs_max and timestep delta "
        f"({suptitle_suffix}; {fixed_note})",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_results(
    results: dict,
    output_dir: str,
    plots: tuple[str, ...] = AVAILABLE_PLOTS,
    fixed_channels: list[int] | None = None,
):
    """results[layer_name][timestep] = {abs_max, mean, std, abs_mean}  (per-channel tensors)
    *plots*: subset of ("heatmap", "curves", "summary") to generate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for layer_name, ts_data in results.items():
        timesteps = sorted(ts_data.keys())
        n_channels = ts_data[timesteps[0]]["abs_max"].shape[0]

        safe_name = layer_name.replace(".", "_").replace("/", "_")

        # ── Plot 1: abs_max heatmap (timestep × channel) ──
        if "heatmap" in plots:
            mat = torch.stack([ts_data[t]["abs_max"] for t in timesteps]).numpy()
            fig, ax = plt.subplots(figsize=(16, 8))
            im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="hot")
            ax.set_xlabel("Channel index")
            ax.set_ylabel("Timestep")
            ax.set_yticks(range(0, len(timesteps), max(1, len(timesteps) // 10)))
            ax.set_yticklabels([f"{timesteps[i]:.0f}" for i in
                                range(0, len(timesteps), max(1, len(timesteps) // 10))])
            ax.set_title(f"{layer_name}\nabs_max per channel across timesteps")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{safe_name}_absmax_heatmap.png"), dpi=150)
            plt.close(fig)

        show_channels = select_plot_channels(
            ts_data, timesteps, n_channels, fixed_channels
        )

        # ── Plot 2: 选几个通道画 abs_max 随 timestep 变化的曲线 ──
        if "curves" in plots:
            fig, ax = plt.subplots(figsize=(14, 6))
            for ch in show_channels:
                values = [ts_data[t]["abs_max"][ch].item() for t in timesteps]
                ax.plot(timesteps, values, label=f"ch {ch}", alpha=0.7)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("abs_max")
            title_suffix = "fixed channels" if fixed_channels else "auto-selected channels"
            ax.set_title(f"{layer_name}\nabs_max of {title_suffix} vs timestep")
            ax.legend(fontsize=7, ncol=3)
            ax.invert_xaxis()
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{safe_name}_absmax_curves.png"), dpi=150)
            plt.close(fig)

        need_channel_mat = any(p in plots for p in ("channel_diff", "channel_step_diff"))
        if need_channel_mat:
            absmax_mat = torch.stack(
                [ts_data[t]["abs_max"][show_channels] for t in timesteps]
            ).numpy()  # (n_timesteps, n_channels)

        # ── Plot 4: 固定通道下，相对 reference timestep 的 abs_max 差异 ──
        if "channel_diff" in plots:
            ref = absmax_mat[0:1, :]
            diff_mat = absmax_mat - ref
            _plot_channel_absmax_grid(
                layer_name=layer_name,
                timesteps=timesteps,
                show_channels=show_channels,
                absmax_mat=absmax_mat,
                delta_mat=diff_mat,
                delta_label=f"Δ vs ts={timesteps[0]:.0f}",
                stat_labels=("span", "rel"),
                suptitle_suffix=f"reference ts={timesteps[0]:.0f}",
                output_path=os.path.join(output_dir, f"{safe_name}_channel_diff.png"),
                fixed_channels=fixed_channels,
            )

        # ── Plot 5: 固定通道下，相对前一个 denoising timestep 的 abs_max 差异 ──
        if "channel_step_diff" in plots:
            step_delta = _compute_prev_step_delta(absmax_mat)
            _plot_channel_absmax_grid(
                layer_name=layer_name,
                timesteps=timesteps,
                show_channels=show_channels,
                absmax_mat=absmax_mat,
                delta_mat=step_delta,
                delta_label="Δ vs prev timestep",
                stat_labels=("max_step", "rel"),
                suptitle_suffix="delta vs previous denoising step",
                output_path=os.path.join(
                    output_dir, f"{safe_name}_channel_step_diff.png"
                ),
                fixed_channels=fixed_channels,
                delta_markers=True,
            )

        # ── Plot 3: 全通道 abs_max 的 mean/max 随 timestep 变化 ──
        if "summary" in plots:
            all_ch_mean = [ts_data[t]["abs_max"].mean().item() for t in timesteps]
            all_ch_max = [ts_data[t]["abs_max"].max().item() for t in timesteps]
            all_ch_std_mean = [ts_data[t]["std"].mean().item() for t in timesteps]

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].plot(timesteps, all_ch_mean, "b-")
            axes[0].set_title("mean(abs_max) over channels")
            axes[0].set_xlabel("Timestep")
            axes[0].invert_xaxis()

            axes[1].plot(timesteps, all_ch_max, "r-")
            axes[1].set_title("max(abs_max) over channels")
            axes[1].set_xlabel("Timestep")
            axes[1].invert_xaxis()

            axes[2].plot(timesteps, all_ch_std_mean, "g-")
            axes[2].set_title("mean(std) over channels")
            axes[2].set_xlabel("Timestep")
            axes[2].invert_xaxis()

            fig.suptitle(layer_name, fontsize=11)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{safe_name}_summary.png"), dpi=150)
            plt.close(fig)

    print(f"Plots saved to {output_dir}/")


def resolve_target_substrings(dit: nn.Module, target_layers: str) -> list[str]:
    if target_layers:
        return [s.strip() for s in target_layers.split(",") if s.strip()]

    block_names = [n for n, _ in dit.named_modules() if ".self_attn.q" in n]
    if not block_names:
        block_names = [n for n, _ in dit.named_modules() if isinstance(_, nn.Linear)][:3]
    n_blocks = len(block_names)
    indices = [0, n_blocks // 2, n_blocks - 1] if n_blocks >= 3 else list(range(n_blocks))
    target_subs = []
    for idx in indices:
        prefix = block_names[idx].rsplit(".", 1)[0]
        target_subs.append(prefix)
    print(f"Auto-selected target prefixes: {target_subs}")
    return target_subs


def _norm_ts(ts) -> float:
    return float(ts)


def pick_snapshot_timesteps(all_timesteps: list, spec: str) -> list[float]:
    """Pick timesteps for 3D activation snapshots."""
    sorted_ts = sorted({_norm_ts(t) for t in all_timesteps})
    if not sorted_ts:
        return []
    if spec.strip():
        requested = [float(x.strip()) for x in spec.split(",") if x.strip()]
        picked = []
        for r in requested:
            best = min(sorted_ts, key=lambda t: abs(t - r))
            if best not in picked:
                picked.append(best)
        return picked
    if len(sorted_ts) <= 3:
        return sorted_ts
    mid = sorted_ts[len(sorted_ts) // 2]
    return [sorted_ts[-1], mid, sorted_ts[0]]


def activation_to_channel_token_grid(
    x: torch.Tensor,
    max_channels: int,
    max_tokens: int,
) -> "np.ndarray":
    """|activation| as (n_channels, n_tokens), downsampled for 3D plot."""
    import numpy as np

    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim > 3:
        x = x.reshape(-1, x.shape[-1])
    # (n_tokens, n_channels)
    grid = x.abs().float().cpu().numpy()

    def _pool_rows(mat, target_rows, axis=0):
        n = mat.shape[axis]
        if n <= target_rows:
            return mat
        step = int(np.ceil(n / target_rows))
        slices = [mat[i::step] for i in range(step)]
        pooled = np.max(np.stack(slices, axis=0), axis=0)
        if axis == 0:
            return pooled[:target_rows]
        return pooled[:, :target_rows]

    def _pool_cols(mat, target_cols):
        n = mat.shape[1]
        if n <= target_cols:
            return mat
        step = int(np.ceil(n / target_cols))
        slices = [mat[:, i::step] for i in range(step)]
        pooled = np.max(np.stack(slices, axis=0), axis=0)
        return pooled[:, :target_cols]

    grid = _pool_rows(grid, max_tokens, axis=0)
    grid = _pool_cols(grid, max_channels)
    return grid.T.astype(np.float32)


def build_channel_timestep_grid(
    ts_data: dict,
    timesteps: list,
    max_channels: int,
) -> "np.ndarray":
    """Stack per-channel abs_max into (n_channels, n_timesteps)."""
    import numpy as np

    mats = [ts_data[t]["abs_max"].numpy() for t in timesteps]
    grid = np.stack(mats, axis=1)
    n_ch = grid.shape[0]
    if n_ch > max_channels:
        step = int(np.ceil(n_ch / max_channels))
        chunks = [grid[i::step] for i in range(step)]
        grid = np.max(np.stack(chunks, axis=0), axis=0)[:max_channels]
    return grid.astype(np.float32)


def _plot_3d_activation_surface(ax, grid, title: str, subtitle: str = ""):
    """Draw 3D surface: X=Channel, Y=Token, Z=|activation|."""
    import numpy as np

    n_channels, n_tokens = grid.shape
    channel_idx = np.arange(n_channels)
    token_idx = np.arange(n_tokens)
    X, Y = np.meshgrid(channel_idx, token_idx, indexing="ij")
    z_max = float(grid.max()) if grid.size else 1.0
    surf = ax.plot_surface(
        X, Y, grid,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        vmin=0.0,
        vmax=max(z_max, 1e-6),
    )
    ax.set_xlabel("Channel", fontsize=9)
    ax.set_ylabel("Token", fontsize=9)
    ax.set_zlabel("|activation|", fontsize=9)
    ax.set_title(f"{title}\n{subtitle}".strip(), fontsize=9)
    ax.view_init(elev=28, azim=-58)
    if z_max > 0:
        ax.set_zlim(0, z_max * 1.05)
    return surf


def plot_activation_3d_comparison(
    baseline_snapshots: dict,
    rotated_snapshots: dict,
    snapshot_timesteps: list[float],
    output_dir: str,
):
    """Only Channel×Token×|activation| 3D plots (X=Channel, Y=Token, Z=value) per timestep."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping activation_3d plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    layers = sorted(set(baseline_snapshots) & set(rotated_snapshots))
    if not layers:
        print("[WARN] No activation snapshots found for activation_3d.")
        return

    for layer_name in layers:
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        base_snaps = baseline_snapshots[layer_name]
        rot_snaps = rotated_snapshots[layer_name]

        for ts in snapshot_timesteps:
            ts_key = _norm_ts(ts)
            if ts_key not in base_snaps or ts_key not in rot_snaps:
                available = sorted(set(base_snaps) & set(rot_snaps))
                if not available:
                    print(f"  [WARN] skip ts={ts_key}: no snapshots for {layer_name}")
                    continue
                ts_key = min(available, key=lambda t: abs(t - ts_key))
                print(f"  [WARN] ts={ts} not found, using closest ts={ts_key}")

            g_base = base_snaps[ts_key]
            g_rot = rot_snaps[ts_key]
            z_peak_base = float(g_base.max())
            z_peak_rot = float(g_rot.max())

            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), subplot_kw={"projection": "3d"})
            _plot_3d_activation_surface(
                axes[0], g_base,
                "Activation (baseline)",
                f"timestep={ts_key:.0f}  max={z_peak_base:.3g}",
            )
            _plot_3d_activation_surface(
                axes[1], g_rot,
                "Activation (rotated)",
                f"timestep={ts_key:.0f}  max={z_peak_rot:.3g}",
            )
            fig.suptitle(
                f"{layer_name}\n3D: Channel (X) × Token (Y) × |activation| (Z)",
                fontsize=11,
            )
            plt.tight_layout()
            ts_tag = str(int(ts_key)) if ts_key == int(ts_key) else f"{ts_key:.1f}".replace(".", "p")
            path = os.path.join(output_dir, f"{safe_name}_activation_3d_ts{ts_tag}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  saved {path}")


def collect_activation_stats(
    dit: nn.Module,
    ts_to_samples: dict[float, list],
    target_subs: list[str],
    device: str,
    snapshot_timesteps: list[float] | None = None,
    max_channels: int = 256,
    max_tokens: int = 128,
) -> tuple[dict, dict]:
    """Run forward passes; return (per-timestep stats, optional channel×token snapshots)."""
    collector = ActivationCollector(dit, target_subs)
    if not collector.hooks:
        raise RuntimeError("No layers matched! Check --target-layers")

    snapshot_set = {_norm_ts(t) for t in (snapshot_timesteps or [])}
    snapshots: dict[str, dict[float, object]] = defaultdict(dict)
    results: dict[str, dict[float, dict[str, torch.Tensor]]] = defaultdict(dict)
    sorted_timesteps = sorted(ts_to_samples.keys(), reverse=True)
    for ts in sorted_timesteps:
        ts_f = _norm_ts(ts)
        ts_samples = ts_to_samples[ts]
        all_stats: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(list)
        snap_saved = False

        for sample in ts_samples:
            collector.clear()
            input_args = [
                a.to(device) if isinstance(a, torch.Tensor) else a
                for a in sample["input_args"]
            ]
            input_kwargs = {}
            for k, v in sample["input_kwargs"].items():
                if isinstance(v, torch.Tensor):
                    input_kwargs[k] = v.to(device)
                else:
                    input_kwargs[k] = v

            with torch.no_grad():
                try:
                    dit(*input_args, **input_kwargs)
                except Exception as e:
                    print(f"  [WARN] Forward failed for ts={ts:.1f}: {e}")
                    continue

            for layer_name, act in collector.data.items():
                stats = compute_channel_stats(act)
                all_stats[layer_name].append(stats)
                if ts_f in snapshot_set and not snap_saved:
                    snapshots[layer_name][ts_f] = activation_to_channel_token_grid(
                        act, max_channels, max_tokens
                    )

            if ts_f in snapshot_set:
                snap_saved = True

        for layer_name, stats_list in all_stats.items():
            if not stats_list:
                continue
            merged = {}
            for key in stats_list[0]:
                stacked = torch.stack([s[key] for s in stats_list])
                if key in ("abs_max",):
                    merged[key] = stacked.amax(dim=0)
                else:
                    merged[key] = stacked.mean(dim=0)
            results[layer_name][ts] = merged

        snap_note = " snapshot" if ts_f in snapshot_set else ""
        print(f"  ts={ts:8.1f}  samples={len(ts_samples)}  "
              f"layers={len(all_stats)}{snap_note}  done")

    collector.remove()
    return dict(results), dict(snapshots)


def _count_channel_changes(
    baseline: torch.Tensor,
    rotated: torch.Tensor,
    rel_tol: float = 1e-6,
) -> dict[str, int | torch.Tensor]:
    diff = rotated - baseline
    eps = baseline.abs().clamp_min(1e-8) * rel_tol
    amplified = diff > eps
    shrunk = diff < -eps
    unchanged = ~(amplified | shrunk)
    rel_change = diff / baseline.abs().clamp_min(1e-8)
    return {
        "amplified": int(amplified.sum().item()),
        "shrunk": int(shrunk.sum().item()),
        "unchanged": int(unchanged.sum().item()),
        "total": int(diff.numel()),
        "diff": diff,
        "rel_change": rel_change,
        "amplified_mask": amplified,
        "shrunk_mask": shrunk,
    }


def _tensor_magnitude_summary(
    rel_change: torch.Tensor,
    amplified_mask: torch.Tensor,
    shrunk_mask: torch.Tensor,
) -> dict[str, float]:
    abs_rel = rel_change.abs()
    summary = {
        "mean_abs_rel_change": float(abs_rel.mean().item()),
        "median_abs_rel_change": float(abs_rel.median().item()),
        "max_abs_rel_change": float(abs_rel.max().item()),
    }
    if amplified_mask.any():
        summary["amplified_mean_rel_change"] = float(rel_change[amplified_mask].mean().item())
        summary["amplified_mean_abs_rel_change"] = float(rel_change[amplified_mask].abs().mean().item())
    else:
        summary["amplified_mean_rel_change"] = 0.0
        summary["amplified_mean_abs_rel_change"] = 0.0
    if shrunk_mask.any():
        summary["shrunk_mean_rel_change"] = float(rel_change[shrunk_mask].mean().item())
        summary["shrunk_mean_abs_rel_change"] = float(rel_change[shrunk_mask].abs().mean().item())
    else:
        summary["shrunk_mean_rel_change"] = 0.0
        summary["shrunk_mean_abs_rel_change"] = 0.0
    return summary


def _compute_channel_ts_metrics(absmax_stack: torch.Tensor) -> dict[str, torch.Tensor]:
    """Metrics describing how each channel varies across timesteps.

    absmax_stack: (n_timesteps, n_channels), timesteps sorted ascending.
    """
    ts_span = absmax_stack.max(dim=0).values - absmax_stack.min(dim=0).values
    ts_peak = absmax_stack.max(dim=0).values.clamp_min(1e-8)
    ts_rel_span = ts_span / ts_peak
    ts_std = absmax_stack.std(dim=0)
    if absmax_stack.shape[0] > 1:
        step_delta = absmax_stack[1:, :] - absmax_stack[:-1, :]
        ts_mean_abs_step = step_delta.abs().mean(dim=0)
        ts_max_abs_step = step_delta.abs().max(dim=0).values
    else:
        zeros = torch.zeros(absmax_stack.shape[1])
        ts_mean_abs_step = zeros
        ts_max_abs_step = zeros
    return {
        "span": ts_span,
        "rel_span": ts_rel_span,
        "std": ts_std,
        "mean_abs_step": ts_mean_abs_step,
        "max_abs_step": ts_max_abs_step,
    }


def _summarize_ts_variation_change(
    base_stack: torch.Tensor,
    rot_stack: torch.Tensor,
    rel_tol: float = 1e-6,
) -> dict:
    base_m = _compute_channel_ts_metrics(base_stack)
    rot_m = _compute_channel_ts_metrics(rot_stack)

    def _delta_summary(base: torch.Tensor, rot: torch.Tensor, name: str) -> dict[str, float | int]:
        delta = rot - base
        ratio = rot / base.clamp_min(1e-8)
        eps = base.abs().clamp_min(1e-8) * rel_tol
        increased = int((delta > eps).sum().item())
        decreased = int((delta < -eps).sum().item())
        unchanged = int((delta.abs() <= eps).sum().item())
        return {
            f"{name}_base_mean": float(base.mean().item()),
            f"{name}_rot_mean": float(rot.mean().item()),
            f"{name}_delta_mean": float(delta.mean().item()),
            f"{name}_ratio_mean": float(ratio.mean().item()),
            f"{name}_ratio_median": float(ratio.median().item()),
            f"{name}_increased": increased,
            f"{name}_decreased": decreased,
            f"{name}_unchanged": unchanged,
        }

    per_ts_step = {}
    if base_stack.shape[0] > 1:
        base_step = (base_stack[1:, :] - base_stack[:-1, :]).abs()
        rot_step = (rot_stack[1:, :] - rot_stack[:-1, :]).abs()
        for i in range(base_step.shape[0]):
            per_ts_step[i] = {
                "base_mean_abs_step": float(base_step[i].mean().item()),
                "rot_mean_abs_step": float(rot_step[i].mean().item()),
                "step_ratio_mean": float(
                    (rot_step[i] / base_step[i].clamp_min(1e-8)).mean().item()
                ),
            }

    return {
        "span": _delta_summary(base_m["span"], rot_m["span"], "span"),
        "rel_span": _delta_summary(base_m["rel_span"], rot_m["rel_span"], "rel_span"),
        "std": _delta_summary(base_m["std"], rot_m["std"], "std"),
        "mean_abs_step": _delta_summary(
            base_m["mean_abs_step"], rot_m["mean_abs_step"], "mean_abs_step"
        ),
        "max_abs_step": _delta_summary(
            base_m["max_abs_step"], rot_m["max_abs_step"], "max_abs_step"
        ),
        "_step_rows": per_ts_step,
    }


def summarize_rotation_channel_changes(
    baseline_results: dict,
    rotated_results: dict,
    rel_tol: float = 1e-6,
) -> dict:
    """Compare all channels before/after rotation using abs_max."""
    summary: dict[str, dict] = {}
    shared_layers = sorted(set(baseline_results) & set(rotated_results))
    for layer_name in shared_layers:
        base_ts = baseline_results[layer_name]
        rot_ts = rotated_results[layer_name]
        timesteps = sorted(set(base_ts) & set(rot_ts))
        if not timesteps:
            continue

        per_ts = {}
        per_ts_magnitude = {}
        ts_amplified = []
        ts_shrunk = []
        ts_unchanged = []
        ts_mean_abs_rel = []
        per_ts_step = {}

        for ts in timesteps:
            counts = _count_channel_changes(
                base_ts[ts]["abs_max"], rot_ts[ts]["abs_max"], rel_tol=rel_tol
            )
            mag = _tensor_magnitude_summary(
                counts["rel_change"], counts["amplified_mask"], counts["shrunk_mask"]
            )
            per_ts[float(ts)] = {
                "amplified": counts["amplified"],
                "shrunk": counts["shrunk"],
                "unchanged": counts["unchanged"],
                "total": counts["total"],
            }
            per_ts_magnitude[float(ts)] = mag
            ts_amplified.append(counts["amplified"])
            ts_shrunk.append(counts["shrunk"])
            ts_unchanged.append(counts["unchanged"])
            ts_mean_abs_rel.append(mag["mean_abs_rel_change"])

        base_stack = torch.stack([base_ts[ts]["abs_max"] for ts in timesteps])
        rot_stack = torch.stack([rot_ts[ts]["abs_max"] for ts in timesteps])
        overall_base = base_stack.amax(dim=0)
        overall_rot = rot_stack.amax(dim=0)
        overall = _count_channel_changes(overall_base, overall_rot, rel_tol=rel_tol)
        overall_magnitude = _tensor_magnitude_summary(
            overall["rel_change"], overall["amplified_mask"], overall["shrunk_mask"]
        )

        ts_variation = _summarize_ts_variation_change(base_stack, rot_stack, rel_tol=rel_tol)
        step_rows = ts_variation.pop("_step_rows")
        if base_stack.shape[0] > 1:
            for i, ts in enumerate(timesteps[:-1]):
                current_ts = float(timesteps[i])
                next_ts = float(timesteps[i + 1])
                row = step_rows[i]
                row["timestep"] = current_ts
                row["prev_timestep"] = next_ts
                row["label"] = f"{next_ts:.0f}->{current_ts:.0f}"
                per_ts_step[current_ts] = row

        summary[layer_name] = {
            "overall": {
                "amplified": overall["amplified"],
                "shrunk": overall["shrunk"],
                "unchanged": overall["unchanged"],
                "total": overall["total"],
                "metric": "max(abs_max) over timesteps",
            },
            "overall_magnitude": overall_magnitude,
            "per_timestep": per_ts,
            "per_timestep_magnitude": per_ts_magnitude,
            "timestep_variation": ts_variation,
            "per_timestep_step_change": per_ts_step,
            "timesteps": timesteps,
            "mean_per_timestep": {
                "amplified": float(sum(ts_amplified) / len(ts_amplified)),
                "shrunk": float(sum(ts_shrunk) / len(ts_shrunk)),
                "unchanged": float(sum(ts_unchanged) / len(ts_unchanged)),
                "mean_abs_rel_change": float(sum(ts_mean_abs_rel) / len(ts_mean_abs_rel)),
            },
        }
    return summary


def print_rotation_compare_summary(summary: dict) -> None:
    for layer_name, info in summary.items():
        overall = info["overall"]
        total = overall["total"]
        print(f"\n[{layer_name}] metric={overall['metric']}")
        for key in ("amplified", "shrunk", "unchanged"):
            count = overall[key]
            pct = 100.0 * count / total if total else 0.0
            print(f"  {key:10s}: {count:5d} / {total} ({pct:5.1f}%)")

        mag = info["overall_magnitude"]
        print("  magnitude (rel_change = (rot-base)/|base|):")
        print(f"    mean|rel|={mag['mean_abs_rel_change']:.4g}, "
              f"median|rel|={mag['median_abs_rel_change']:.4g}, "
              f"max|rel|={mag['max_abs_rel_change']:.4g}")
        print(f"    amplified: mean_rel={mag['amplified_mean_rel_change']:.4g}, "
              f"mean|rel|={mag['amplified_mean_abs_rel_change']:.4g}")
        print(f"    shrunk   : mean_rel={mag['shrunk_mean_rel_change']:.4g}, "
              f"mean|rel|={mag['shrunk_mean_abs_rel_change']:.4g}")

        mean_ts = info["mean_per_timestep"]
        print(f"  per-ts avg : amplified={mean_ts['amplified']:.1f}, "
              f"shrunk={mean_ts['shrunk']:.1f}, unchanged={mean_ts['unchanged']:.1f}, "
              f"mean|rel|={mean_ts['mean_abs_rel_change']:.4g}")

        tv = info["timestep_variation"]
        print("  cross-timestep variation change (same channel, across timesteps):")
        for metric in ("span", "rel_span", "std", "mean_abs_step"):
            m = tv[metric]
            print(
                f"    {metric:14s}: base={m[f'{metric}_base_mean']:.4g}, "
                f"rot={m[f'{metric}_rot_mean']:.4g}, "
                f"ratio_mean={m[f'{metric}_ratio_mean']:.4g}, "
                f"inc/dec/unch={m[f'{metric}_increased']}/"
                f"{m[f'{metric}_decreased']}/{m[f'{metric}_unchanged']}"
            )


def plot_rotation_compare(summary: dict, output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available, skipping rotation_compare plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for layer_name, info in summary.items():
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        timesteps = info["timesteps"]
        per_ts = info["per_timestep"]
        per_ts_mag = info["per_timestep_magnitude"]
        per_ts_step = info.get("per_timestep_step_change", {})
        amplified = [per_ts[float(ts)]["amplified"] for ts in timesteps]
        shrunk = [per_ts[float(ts)]["shrunk"] for ts in timesteps]
        unchanged = [per_ts[float(ts)]["unchanged"] for ts in timesteps]
        mean_abs_rel = [per_ts_mag[float(ts)]["mean_abs_rel_change"] for ts in timesteps]
        overall = info["overall"]
        tv = info["timestep_variation"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        labels = ["amplified", "shrunk", "unchanged"]
        counts = [overall[k] for k in labels]
        colors = ["#d62728", "#1f77b4", "#7f7f7f"]
        axes[0, 0].bar(labels, counts, color=colors)
        axes[0, 0].set_ylabel("channel count")
        axes[0, 0].set_title(f"Overall counts ({overall['metric']})")
        for i, v in enumerate(counts):
            pct = 100.0 * v / overall["total"] if overall["total"] else 0.0
            axes[0, 0].text(i, v, f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

        axes[0, 1].plot(timesteps, amplified, "o-", color="#d62728", label="amplified", markersize=3)
        axes[0, 1].plot(timesteps, shrunk, "o-", color="#1f77b4", label="shrunk", markersize=3)
        axes[0, 1].plot(timesteps, unchanged, "o-", color="#7f7f7f", label="unchanged", markersize=3)
        axes[0, 1].set_xlabel("Timestep")
        axes[0, 1].set_ylabel("channel count")
        axes[0, 1].set_title("Per-timestep channel count change")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].invert_xaxis()

        axes[1, 0].plot(
            timesteps, mean_abs_rel, "o-", color="#9467bd", markersize=3, label="mean |rel_change|"
        )
        mag = info["overall_magnitude"]
        axes[1, 0].axhline(
            mag["amplified_mean_abs_rel_change"], color="#d62728", linestyle="--", alpha=0.7,
            label=f"overall amp mean|rel|={mag['amplified_mean_abs_rel_change']:.3g}",
        )
        axes[1, 0].axhline(
            mag["shrunk_mean_abs_rel_change"], color="#1f77b4", linestyle="--", alpha=0.7,
            label=f"overall shrink mean|rel|={mag['shrunk_mean_abs_rel_change']:.3g}",
        )
        axes[1, 0].set_xlabel("Timestep")
        axes[1, 0].set_ylabel("relative magnitude")
        axes[1, 0].set_title("Per-timestep mean |rel_change|")
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].invert_xaxis()

        metric_names = ["span", "rel_span", "std", "mean_abs_step"]
        x = np.arange(len(metric_names))
        width = 0.35
        base_vals = [tv[m][f"{m}_base_mean"] for m in metric_names]
        rot_vals = [tv[m][f"{m}_rot_mean"] for m in metric_names]
        axes[1, 1].bar(x - width / 2, base_vals, width, label="baseline", color="#1f77b4")
        axes[1, 1].bar(x + width / 2, rot_vals, width, label="rotated", color="#d62728")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metric_names, rotation=20, ha="right")
        axes[1, 1].set_title("Cross-timestep variation (channel-mean)")
        axes[1, 1].legend(fontsize=8)

        fig.suptitle(
            f"{layer_name}\nRotation vs baseline: counts and magnitudes",
            fontsize=11,
        )
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{safe_name}_rotation_compare.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  saved {out_path}")

        if per_ts_step:
            step_ts = sorted(per_ts_step.keys())
            base_step = [per_ts_step[ts]["base_mean_abs_step"] for ts in step_ts]
            rot_step = [per_ts_step[ts]["rot_mean_abs_step"] for ts in step_ts]
            step_ratio = [per_ts_step[ts]["step_ratio_mean"] for ts in step_ts]

            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
            axes2[0].plot(step_ts, base_step, "o-", color="#1f77b4", label="baseline", markersize=3)
            axes2[0].plot(step_ts, rot_step, "o-", color="#d62728", label="rotated", markersize=3)
            axes2[0].set_xlabel("Timestep (current)")
            axes2[0].set_ylabel("mean |step delta| over channels")
            axes2[0].set_title("Adjacent-timestep jump magnitude")
            axes2[0].legend(fontsize=8)
            axes2[0].invert_xaxis()

            axes2[1].plot(step_ts, step_ratio, "o-", color="#2ca02c", markersize=3)
            axes2[1].axhline(1.0, color="#7f7f7f", linestyle=":", alpha=0.8)
            axes2[1].set_xlabel("Timestep (current)")
            axes2[1].set_ylabel("rot / base")
            axes2[1].set_title("Step-jump ratio after rotation")
            axes2[1].invert_xaxis()

            fig2.suptitle(
                f"{layer_name}\nHow rotation changes per-channel timestep differences",
                fontsize=11,
            )
            plt.tight_layout()
            out_path2 = os.path.join(output_dir, f"{safe_name}_rotation_ts_variation.png")
            fig2.savefig(out_path2, dpi=150)
            plt.close(fig2)
            print(f"  saved {out_path2}")


def save_rotation_compare_summary(summary: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "rotation_compare_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    pt_path = os.path.join(output_dir, "rotation_compare_summary.pt")
    torch.save(summary, pt_path)
    print(f"Rotation compare summary saved to {json_path}")
    return json_path


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analyze per-channel activation distributions by timestep")
    parser.add_argument("--calib-dir", required=True, help="Path to calibration .pt files")
    parser.add_argument("--model-path", default="/data1/lyf/Lab/DiffSynth-Studio/models",
                        help="Path to model weights root")
    parser.add_argument("--output-dir", default="analysis_output", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of sample_ids to use")
    parser.add_argument("--target-layers", type=str, default="",
                        help="Comma-separated layer name substrings. Empty = auto-select")
    parser.add_argument("--guidance", type=int, default=0,
                        help="Which guidance mode to analyze (0=uncond, 1=cond)")
    parser.add_argument("--plots", type=str,
                        default="heatmap,curves,summary,channel_diff,channel_step_diff",
                        help="Comma-separated: heatmap, curves, summary, channel_diff, "
                             "channel_step_diff, rotation_compare, activation_3d")
    parser.add_argument("--fixed-channels", type=str, default="",
                        help="Comma-separated channel indices (typically 12) for curves/"
                             "channel_diff/channel_step_diff. Use the same values with/without "
                             "--rotation for fair comparison.")
    parser.add_argument("--rotation", action="store_true",
                        help="Apply Hadamard rotation before collecting activations "
                             "(to verify if rotation flattens per-channel distributions)")
    parser.add_argument("--compare-rotation", action="store_true",
                        help="Run baseline + rotated collection in one pass, then count "
                             "amplified/shrunk channels over all channels")
    parser.add_argument("--compare-with", type=str, default="",
                        help="Path to baseline activation_stats_by_timestep.pt; compare it "
                             "with the current run (use with --rotation for rotated side)")
    parser.add_argument("--change-rel-tol", type=float, default=1e-6,
                        help="Relative tolerance when classifying channel increase/decrease")
    parser.add_argument("--activation-3d-timesteps", type=str, default="",
                        help="Timesteps for Channel×Token 3D plots (comma-separated). "
                             "Default: highest, middle, lowest.")
    parser.add_argument("--activation-3d-max-channels", type=int, default=256,
                        help="Max channels in downsampled 3D activation grid")
    parser.add_argument("--activation-3d-max-tokens", type=int, default=128,
                        help="Max tokens in downsampled 3D activation grid")
    parser.add_argument("--baseline-snapshots", type=str, default="",
                        help="Optional activation_snapshots.pt for baseline when using --compare-with")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    if args.compare_rotation and args.rotation:
        parser.error("Use either --compare-rotation or --rotation, not both.")
    if args.compare_rotation and args.compare_with:
        parser.error("Use either --compare-rotation or --compare-with, not both.")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load calibration samples
    print("=" * 60)
    print("Loading calibration files...")
    all_samples = load_calib_files(args.calib_dir, num_samples=args.num_samples)
    samples = [s for s in all_samples if s.get("guidance", 0) == args.guidance]
    print(f"Filtered to guidance={args.guidance}: {len(samples)} samples")

    ts_to_samples: dict[float, list] = defaultdict(list)
    for s in samples:
        ts = s["input_kwargs"]["timestep"].float().item()
        ts_to_samples[ts].append(s)
    print(f"Unique timesteps: {len(ts_to_samples)}")

    selected_plots = tuple(p.strip() for p in args.plots.split(",") if p.strip())
    need_activation_3d = "activation_3d" in selected_plots
    snapshot_timesteps = pick_snapshot_timesteps(
        list(ts_to_samples.keys()), args.activation_3d_timesteps
    )
    if need_activation_3d:
        print(f"activation_3d snapshot timesteps: {snapshot_timesteps}")

    def _load_model_and_collect(apply_rotation: bool, collect_snapshots: bool):
        print("=" * 60)
        label = "rotated" if apply_rotation else "baseline"
        print(f"Loading model ({label})...")
        pipeline = build_pipeline(args.model_path, device=args.device)
        dit = pipeline.dit
        dit.eval()
        target_subs = resolve_target_substrings(dit, args.target_layers)
        rotation_hooks = []
        if apply_rotation:
            print("Applying Hadamard rotation to DiT Linear layers...")
            rotation_hooks = apply_hadamard_rotation(dit, device=args.device)
        print("=" * 60)
        print(f"Running forward passes ({label})...")
        snap_ts = snapshot_timesteps if collect_snapshots else []
        results, snapshots = collect_activation_stats(
            dit,
            ts_to_samples,
            target_subs,
            args.device,
            snapshot_timesteps=snap_ts,
            max_channels=args.activation_3d_max_channels,
            max_tokens=args.activation_3d_max_tokens,
        )
        for h in rotation_hooks:
            h.remove()
        del pipeline, dit
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return results, snapshots

    baseline_results = None
    rotated_results = None
    baseline_snapshots: dict = {}
    rotated_snapshots: dict = {}

    if args.compare_rotation:
        baseline_results, baseline_snapshots = _load_model_and_collect(
            apply_rotation=False, collect_snapshots=need_activation_3d
        )
        baseline_path = os.path.join(args.output_dir, "activation_stats_by_timestep.pt")
        torch.save(baseline_results, baseline_path)
        print(f"Baseline stats saved to {baseline_path}")
        if need_activation_3d:
            snap_path = os.path.join(args.output_dir, "activation_snapshots_baseline.pt")
            torch.save(baseline_snapshots, snap_path)
            print(f"Baseline snapshots saved to {snap_path}")
        rotated_results, rotated_snapshots = _load_model_and_collect(
            apply_rotation=True, collect_snapshots=need_activation_3d
        )
        if need_activation_3d:
            snap_path = os.path.join(args.output_dir, "activation_snapshots_rotated.pt")
            torch.save(rotated_snapshots, snap_path)
            print(f"Rotated snapshots saved to {snap_path}")
        results = rotated_results
        save_tag = "_rotated"
    elif args.compare_with:
        if not os.path.isfile(args.compare_with):
            raise FileNotFoundError(f"Baseline stats not found: {args.compare_with}")
        print("=" * 60)
        print(f"Loading baseline stats from {args.compare_with}")
        baseline_results = torch.load(args.compare_with, weights_only=False)
        if args.baseline_snapshots and os.path.isfile(args.baseline_snapshots):
            baseline_snapshots = torch.load(args.baseline_snapshots, weights_only=False)
            print(f"Loaded baseline snapshots from {args.baseline_snapshots}")
        elif need_activation_3d:
            print("[WARN] No --baseline-snapshots: Channel×Token 3D will only show rotated side "
                  "or skip missing timesteps.")
        results, rotated_snapshots = _load_model_and_collect(
            apply_rotation=True, collect_snapshots=need_activation_3d
        )
        rotated_results = results
        save_tag = "_rotated"
        if need_activation_3d:
            snap_path = os.path.join(args.output_dir, "activation_snapshots_rotated.pt")
            torch.save(rotated_snapshots, snap_path)
            print(f"Rotated snapshots saved to {snap_path}")
        if not args.rotation:
            raise ValueError("--compare-with expects the current run to use --rotation.")
    else:
        print("=" * 60)
        results, snapshots = _load_model_and_collect(
            apply_rotation=args.rotation,
            collect_snapshots=need_activation_3d,
        )
        save_tag = "_rotated" if args.rotation else ""
        if need_activation_3d:
            tag = "rotated" if args.rotation else "baseline"
            snap_path = os.path.join(args.output_dir, f"activation_snapshots_{tag}.pt")
            torch.save(snapshots, snap_path)
            print(f"Snapshots saved to {snap_path}")
            if args.rotation:
                rotated_results, rotated_snapshots = results, snapshots
            else:
                baseline_results, baseline_snapshots = results, snapshots

    # Save raw data for the primary run
    save_path = os.path.join(args.output_dir, f"activation_stats_by_timestep{save_tag}.pt")
    torch.save(dict(results), save_path)
    print(f"Raw stats saved to {save_path}")

    run_rotation_compare = "rotation_compare" in selected_plots
    standard_plot_types = set(AVAILABLE_PLOTS) - {"activation_3d", "rotation_compare"}
    run_standard_plots = bool(set(selected_plots) & standard_plot_types)

    if run_rotation_compare and baseline_results is not None and rotated_results is not None:
        print("=" * 60)
        print("Comparing rotation vs baseline (all channels, abs_max)...")
        rotation_summary = summarize_rotation_channel_changes(
            baseline_results,
            rotated_results,
            rel_tol=args.change_rel_tol,
        )
        print_rotation_compare_summary(rotation_summary)
        save_rotation_compare_summary(rotation_summary, args.output_dir)
        print("=" * 60)
        print("Generating rotation_compare plots...")
        plot_rotation_compare(rotation_summary, args.output_dir)

    if need_activation_3d:
        if not baseline_snapshots or not rotated_snapshots:
            print("[WARN] activation_3d needs snapshots from --compare-rotation "
                  "(or --compare-with + --baseline-snapshots + --rotation).")
        else:
            print("=" * 60)
            print("Generating activation_3d (Channel×Token×|activation| only)...")
            plot_activation_3d_comparison(
                baseline_snapshots,
                rotated_snapshots,
                snapshot_timesteps,
                args.output_dir,
            )

    if run_standard_plots:
        print("=" * 60)
        print("Generating plots...")
        fixed_channels = None
        if args.fixed_channels.strip():
            first_layer = next(iter(results))
            n_channels = results[first_layer][sorted(results[first_layer].keys())[0]]["abs_max"].shape[0]
            fixed_channels = parse_fixed_channels(args.fixed_channels, n_channels)
            print(f"Using fixed channels ({len(fixed_channels)}): {fixed_channels}")
        elif any(p in selected_plots for p in ("curves", "channel_diff", "channel_step_diff")):
            print("[WARN] --fixed-channels not set; channel plots will auto-select channels, "
                  "which may differ between rotation and non-rotation runs.")

        plot_results(
            dict(results),
            args.output_dir,
            plots=tuple(p for p in selected_plots if p in standard_plot_types),
            fixed_channels=fixed_channels,
        )
    print("Done!")


if __name__ == "__main__":
    main()
