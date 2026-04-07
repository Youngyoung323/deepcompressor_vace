# -*- coding: utf-8 -*-
"""Collect calibration dataset for Wan2.1-T2V-1.3B model.

Pipeline is created via WanVideoPipeline.from_pretrained() with T2V-1.3B weights.
Dataset is loaded from Ditto-1M source_video_captions_sorted.json, filtered to
only include samples whose video directory exists locally.

Usage:
    python -m deepcompressor.app.diffusion.dataset.collect.calib_wan_t2v_1_3b
"""

import json
import os
import sys

sys.path.insert(0, "/data1/lyf/Lab/VACE/deepcompressor_vace")

import torch
from tqdm import tqdm

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

from deepcompressor.utils.common import hash_str_to_int, tree_map

from deepcompressor.app.diffusion.dataset.collect.utils import ModelFnCollectHook


def process(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)


def load_dataset(
    caption_json_path: str,
    video_root: str,
    max_samples: int = -1,
) -> list[dict]:
    """Load Ditto-1M captions and filter by locally available video directories.

    Args:
        caption_json_path: Path to source_video_captions_sorted.json.
        video_root: Path to videos/source/source/ containing subdirectories.
        max_samples: Maximum number of samples to load (-1 for all).

    Returns:
        List of dicts with keys: sample_id, prompt, video_path.
    """
    available_dirs = set(os.listdir(video_root))

    with open(caption_json_path, "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    samples = []
    for item in all_entries:
        rel_path = item["path"]
        dir_name = rel_path.split("/")[0]
        if dir_name not in available_dirs:
            continue

        video_full_path = os.path.join(video_root, rel_path)
        if not os.path.exists(video_full_path):
            continue

        file_stem = os.path.splitext(os.path.basename(rel_path))[0]
        sample_id = f"{dir_name}_{file_stem}"

        samples.append({
            "sample_id": sample_id,
            "prompt": item["caption"],
            "video_path": video_full_path,
        })

        if 0 < max_samples <= len(samples):
            break

    return samples


def collect(
    pipeline: WanVideoPipeline,
    samples: list[dict],
    output_root: str,
    num_steps: int,
    base_pipeline_kwargs: dict,
):
    """Run T2V inference over the dataset, capture dit activations, and save them."""
    samples_dirpath = os.path.join(output_root, "samples")
    caches_dirpath = os.path.join(output_root, "caches")
    os.makedirs(samples_dirpath, exist_ok=True)
    os.makedirs(caches_dirpath, exist_ok=True)
    caches: list[dict] = []

    hook = ModelFnCollectHook(model_fn=pipeline.model_fn, caches=caches)
    pipeline.model_fn = hook

    print(f"In total {len(samples)} samples")
    for idx, sample in enumerate(
        tqdm(samples, desc="T2V Calib", leave=False, dynamic_ncols=True),
        start=1,
    ):
        # 固定 6 位序号文件名：000001, 000002, ...（与样本在列表中的顺序一致）
        seq_id = f"{idx:06d}"
        original_id = sample["sample_id"]
        prompt = sample["prompt"]
        seed = hash_str_to_int(seq_id)

        cache_prefix = os.path.join(caches_dirpath, f"{seq_id}-00000-0.pt")
        if os.path.exists(cache_prefix):
            print(f"Skipping {seq_id} ({original_id}) (cache exists)")
            continue

        result_video = pipeline(
            prompt=prompt,
            seed=seed,
            **base_pipeline_kwargs,
        )

        num_guidances = len(caches) // num_steps
        assert (
            len(caches) == num_steps * num_guidances
        ), f"Unexpected number of caches: {len(caches)} != {num_steps} * {num_guidances}"

        save_video(
            result_video,
            os.path.join(samples_dirpath, f"{seq_id}.mp4"),
            fps=15,
            quality=5,
        )

        for s in range(num_steps):
            for g in range(num_guidances):
                c = caches[s * num_guidances + g]
                c["sample_id"] = seq_id
                c["original_sample_id"] = original_id
                c["step"] = s
                c["guidance"] = g
                c = tree_map(lambda x: process(x), c)
                torch.save(
                    c,
                    os.path.join(caches_dirpath, f"{seq_id}-{s:05d}-{g}.pt"),
                )
        caches.clear()


if __name__ == "__main__":
    # ─── Model ───
    torch_dtype = torch.bfloat16
    device = "cuda"
    model_base = "/data1/lyf/Lab/DiffSynth-Studio/models"

    model_configs = [
        ModelConfig(
            path=os.path.join(
                model_base,
                "Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            ),
        ),
        ModelConfig(
            path=os.path.join(
                model_base,
                "DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
                "models_t5_umt5-xxl-enc-bf16.safetensors",
            ),
        ),
        ModelConfig(
            path=os.path.join(
                model_base,
                "DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
                "Wan2.1_VAE.safetensors",
            ),
        ),
    ]
    tokenizer_config = ModelConfig(
        path=os.path.join(model_base, "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
    )

    pipeline = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    # ─── Dataset ───
    caption_json_path = (
        "/data1/lyf/Lab/Video_dataset/Ditto-1M/"
        "source_video_captions/source_video_captions_sorted.json"
    )
    video_root = "/data1/lyf/Lab/Video_dataset/Ditto-1M/videos/source/source"
    num_samples = 128

    # ─── Generation parameters ───
    num_steps = 50
    height = 480
    width = 832
    num_frames = 81
    cfg_scale = 5.0
    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )
    tiled = True
    sigma_shift = 5.0

    # ─── Output ───
    output_root = "/data1/lyf/Lab/VACE/deepcompressor_vace/examples/diffusion/datasets"
    dataset_name = "Ditto-1M"
    pipeline_name = "Wan2.1-T2V-1.3B-50steps"

    collect_dirpath = os.path.join(
        output_root,
        str(torch_dtype),
        pipeline_name,
        dataset_name,
        f"s{num_samples}",
    )
    print(f"Saving caches to {collect_dirpath}")

    samples = load_dataset(
        caption_json_path, video_root, max_samples=num_samples,
    )
    print(f"Loaded {len(samples)} samples (filtered by available videos)")

    base_pipeline_kwargs: dict = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_steps,
        "cfg_scale": cfg_scale,
        "negative_prompt": negative_prompt,
        "tiled": tiled,
        "sigma_shift": sigma_shift,
    }

    os.makedirs(collect_dirpath, exist_ok=True)
    collect(
        pipeline,
        samples=samples,
        output_root=collect_dirpath,
        num_steps=num_steps,
        base_pipeline_kwargs=base_pipeline_kwargs,
    )
