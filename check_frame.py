"""
对照：同一 benchmark 下
  - samples/*.mp4 的 OpenCV 帧数
  - caches/*.pt 里 latent 的时间维 F（与 calib_wan 保存的 input_args[0] 一致）

命名约定（与当前数据一致）：
  视频 000042.mp4  <->  校准 000042-xxxxx-y.pt（第一段为 video_id）
"""

import os
import glob
from pathlib import Path
from collections import defaultdict

import cv2
import torch

# 配置路径：samples 放 mp4，同级目录 caches 放 .pt
VIDEO_DIR = "/data1/lyf/Lab/VACE/deepcompressor_vace/examples/diffusion/datasets/torch.bfloat16/Wan2.1-VACE-14B-lora-20steps/VACE-benchmark-real/s120/samples"
# 若与 VIDEO_DIR 同级，可自动推导；也可手动覆盖
CACHE_DIR = os.path.join(os.path.dirname(VIDEO_DIR.rstrip(os.sep)), "caches")

VIDEO_EXTENSIONS = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.gif"]


def get_video_frame_count(video_path: str) -> int | None:
    """使用 OpenCV 获取视频帧数。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if count <= 0:
        count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            count += 1

    cap.release()
    return int(count)


def _parse_video_id_from_pt_stem(stem: str) -> str | None:
    """例如 000042-00019-0 -> 000042"""
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    return parts[0]


def scan_calib_latent_f(cache_dir: str) -> dict[str, dict]:
    """
    扫描 caches 下所有 .pt，按 video_id 聚合 latent 形状。

    Returns:
        video_id -> {
            "F_set": set[int],           # 出现过的 latent 时间维 F
            "shapes": dict[tuple, int],  # 完整 (B,C,F,H,W) -> 文件数
            "n_pt": int,
        }
    """
    out: dict[str, dict] = defaultdict(
        lambda: {"F_set": set(), "shapes": defaultdict(int), "n_pt": 0}
    )
    if not os.path.isdir(cache_dir):
        return {}

    for fpath in sorted(glob.glob(os.path.join(cache_dir, "*.pt"))):
        vid = _parse_video_id_from_pt_stem(Path(fpath).stem)
        if vid is None:
            continue
        try:
            data = torch.load(fpath, weights_only=False, map_location="cpu")
            if not isinstance(data, dict) or "input_args" not in data or not data["input_args"]:
                continue
            latent = data["input_args"][0]
            if not hasattr(latent, "shape") or len(latent.shape) < 5:
                continue
            # (B, C, F, H, W)
            f_dim = int(latent.shape[2])
            tup = tuple(latent.shape)
        except Exception:
            continue

        e = out[vid]
        e["F_set"].add(f_dim)
        e["shapes"][tup] += 1
        e["n_pt"] += 1

    # set 转普通结构便于打印
    for vid in out:
        out[vid]["F_set"] = sorted(out[vid]["F_set"])
    return dict(out)


def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"错误: 视频目录不存在 -> {VIDEO_DIR}")
        return

    print(f"视频目录: {VIDEO_DIR}")
    print(f"校准目录: {CACHE_DIR}")
    if not os.path.isdir(CACHE_DIR):
        print(f"警告: 校准目录不存在，将跳过 .pt 对照。\n")

    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
        video_files.extend(glob.glob(os.path.join(VIDEO_DIR, ext.upper())))
    video_files.sort()

    # ---------- mp4 帧数 ----------
    frame_counts_hist: dict[int, list[str]] = {}
    video_id_to_frames: dict[str, int] = {}

    print(f"\n找到 {len(video_files)} 个视频文件，检查 OpenCV 帧数...\n")
    print(f"{'video_id':<10} | {'mp4 文件名':<22} | {'帧数':<8} | 状态")
    print("-" * 70)

    for v_path in video_files:
        fname = os.path.basename(v_path)
        stem = Path(fname).stem
        frames = get_video_frame_count(v_path)
        if frames is None:
            print(f"{stem:<10} | {fname:<22} | {'?':<8} | 无法读取")
            continue
        video_id_to_frames[stem] = frames
        if frames not in frame_counts_hist:
            frame_counts_hist[frames] = []
        frame_counts_hist[frames].append(fname)
        print(f"{stem:<10} | {fname:<22} | {frames:<8} | OK")

    print("-" * 70)
    print("\n=== mp4 帧数统计 ===")
    for fc, names in sorted(frame_counts_hist.items(), key=lambda x: -len(x[1])):
        print(f"  {fc} 帧: {len(names)} 个视频")

    # ---------- .pt latent F ----------
    calib = scan_calib_latent_f(CACHE_DIR)
    print("\n=== 校准 .pt 与 mp4 对照（按 video_id）===")
    if not calib:
        print("  无可用 .pt 或未解析到 input_args[0]。")
    else:
        # 并集：出现在 mp4 或 pt 中的 id
        all_ids = sorted(set(video_id_to_frames.keys()) | set(calib.keys()))
        print(
            f"{'video_id':<10} | {'mp4帧数':<8} | {'latent F':<16} | "
            f"{'.pt数量':<8} | 说明"
        )
        print("-" * 85)

        n_mismatch_f = 0
        n_only_video = 0
        n_only_pt = 0

        for vid in all_ids:
            nf = video_id_to_frames.get(vid)
            ce = calib.get(vid)
            if ce is None:
                print(f"{vid:<10} | {str(nf):<8} | {'(无.pt)':<16} | {'0':<8} | 仅有视频")
                n_only_video += 1
                continue
            if nf is None:
                print(f"{vid:<10} | {'(无mp4)':<8} | {str(ce['F_set']):<16} | {ce['n_pt']:<8} | 仅有校准")
                n_only_pt += 1
                continue

            f_str = str(ce["F_set"])
            if len(ce["F_set"]) > 1:
                note = "同一视频多种 latent 形状（异常）"
                n_mismatch_f += 1
            else:
                note = "OK"

            print(f"{vid:<10} | {nf:<8} | {f_str:<16} | {ce['n_pt']:<8} | {note}")

        print("-" * 85)
        print(
            f"汇总: 有 mp4+pt 对照 {len(all_ids) - n_only_video - n_only_pt} 个 id；"
            f"仅视频 {n_only_video}；仅.pt {n_only_pt}；"
            f"同一 id 多种 F {n_mismatch_f}"
        )

        # 按 latent F 汇总「哪些 video_id」
        by_f: dict[int, list[str]] = defaultdict(list)
        for vid, ce in calib.items():
            for f in ce["F_set"]:
                by_f[f].append(vid)
        print("\n=== 按 latent 时间维 F 汇总 video_id（用于解释 21/22 来源）===")
        for f in sorted(by_f.keys()):
            ids = sorted(by_f[f], key=lambda x: int(x))
            print(f"  F={f}: {len(ids)} 个视频 id，例如: {ids[:8]}{'...' if len(ids) > 8 else ''}")

    print("\n说明: latent 的 F 由 VAE 时间下采样与对齐决定，一般不等于 mp4 的帧数；")
    print("      若多数 mp4 帧数相同而 F 仍有 21/22 两种，通常是个别视频编码/预处理导致有效 T 不同。")


if __name__ == "__main__":
    main()
