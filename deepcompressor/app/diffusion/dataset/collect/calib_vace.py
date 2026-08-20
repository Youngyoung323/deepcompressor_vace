# -*- coding: utf-8 -*-
"""Collect calibration dataset for Wan Video (VACE) model.

Pipeline is created explicitly via WanVideoPipeline.from_pretrained()
Dataset is loaded from VACE-Benchmark real.txt
"""

import glob
import json
import os
import sys
sys.path.insert(0, "/data1/lyf/Lab/VACE/deepcompressor_vace")

import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video, VideoData

from deepcompressor.utils.common import hash_str_to_int, tree_map

from deepcompressor.app.diffusion.dataset.collect.utils import ModelFnCollectHook


def process(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)


def load_dataset(index_path: str, process_data_root: str, max_samples: int = -1) -> list[dict]:
    """Load VACE-Benchmark dataset from JSONL index file.

    Each line in the index file is a JSON object with fields:
        source, sample_id, task, subtask, raw_video, src_video,
        src_mask, src_ref_images, en_prompt, zh_prompt
    """
    samples = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample = {
                "sample_id": item["sample_id"],
                "task": item["task"],
                "subtask": item["subtask"],
                "prompt": item["en_prompt"],
                "src_video": item["src_video"],
                "src_mask": item["src_mask"],
                "src_ref_images": [
                    os.path.join(process_data_root, os.path.basename(p))
                    for p in item.get("src_ref_images", [])
                ],
            }
            samples.append(sample)
            if 0 < max_samples <= len(samples):
                break
    return samples


def build_pipeline_kwargs_for_sample(
    sample: dict,
    height: int,
    width: int,
) -> dict:
    """Build per-sample VACE kwargs based on task / subtask.

    Different VACE tasks require different combinations of
    vace_video, vace_video_mask, and vace_reference_image.
    """
    kwargs: dict = {}
    task = sample["task"]
    subtask = sample["subtask"]
    src_video_path = sample["src_video"]
    src_mask_path = sample["src_mask"]
    ref_images = sample["src_ref_images"]

    if task in ("MV2V", "V2V", "T2V"):
        # MV2V (inpainting / outpainting / frame-level tasks) and
        # V2V (depth / flow / gray / layout / pose / scribble)
        # all provide a source (condition) video and a mask.
        kwargs["vace_video"] = VideoData(src_video_path, height=height, width=width)
        kwargs["vace_video_mask"] = VideoData(src_mask_path, height=height, width=width)
    elif task == "R2V":
        # Reference-image-to-video: provide the reference image
        if ref_images:
            kwargs["vace_reference_image"] = Image.open(ref_images[0]).resize((width, height))
    else:
        print(f"Warning: unknown task '{task}', skipping VACE-specific kwargs")

    return kwargs


def collect(
    pipeline: WanVideoPipeline,
    samples: list[dict],
    output_root: str,
    num_steps: int,
    base_pipeline_kwargs: dict,
    height: int,
    width: int,
):
    """Run inference over the dataset, capture dit activations, and save them."""
    samples_dirpath = os.path.join(output_root, "samples")
    caches_dirpath = os.path.join(output_root, "caches")
    os.makedirs(samples_dirpath, exist_ok=True)
    os.makedirs(caches_dirpath, exist_ok=True)
    caches: list[dict] = []

    # In WanVideo VACE, model_fn_wan_video accesses dit's submodules directly
    # (blocks, head, etc.) without calling dit.forward(), so a forward_hook on
    # dit would never fire.  Wrap model_fn instead to intercept every call.
    hook = ModelFnCollectHook(model_fn=pipeline.model_fn, caches=caches)
    pipeline.model_fn = hook

    print(f"In total {len(samples)} samples")
    for sample in tqdm(samples, desc="Data", leave=False, dynamic_ncols=True):
        sample_id = sample["sample_id"]
        prompt = sample["prompt"]
        seed = hash_str_to_int(sample_id)

        per_sample_kwargs = build_pipeline_kwargs_for_sample(sample, height, width)
        pipeline_kwargs = {**base_pipeline_kwargs, **per_sample_kwargs}

        result_video = pipeline(
            prompt=prompt,
            seed=seed,
            **pipeline_kwargs,
        )

        num_guidances = len(caches) // num_steps
        assert (
            len(caches) == num_steps * num_guidances
        ), f"Unexpected number of caches: {len(caches)} != {num_steps} * {num_guidances}"

        save_video(result_video, os.path.join(samples_dirpath, f"{sample_id}.mp4"), fps=15, quality=5)

        for s in range(num_steps):
            for g in range(num_guidances):
                c = caches[s * num_guidances + g]
                c["sample_id"] = sample_id
                c["step"] = s
                c["guidance"] = g
                c = tree_map(lambda x: process(x), c)
                torch.save(c, os.path.join(caches_dirpath, f"{sample_id}-{s:05d}-{g}.pt"))
        caches.clear()


if __name__ == "__main__":
    torch_dtype = torch.bfloat16
    device = "cuda"
    model_base = "/data1/lyf/Lab/DiffSynth-Studio/models"
    model_configs = [
        ModelConfig(
            path=sorted(glob.glob(os.path.join(
                model_base, "Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model*.safetensors"
            ))),
        ),
        ModelConfig(
            path=os.path.join(
                model_base, "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
            ),
        ),
        ModelConfig(
            path=os.path.join(
                model_base, "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"
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

    pipeline.load_lora(pipeline.dit, "/data1/lyf/Lab/DiffSynth-Studio/lora/wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors", alpha=1)

    index_path = "/data1/lyf/video_data/json/real_fix.txt"
    process_data_root = "/data1/lyf/video_data/process_data"
    num_samples = 120  

    num_steps = 20               
    height = 480                 
    width = 832                  
    num_frames = 81              
    cfg_scale = 5.0              
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    tiled = True                
    sigma_shift = 5.0            

    output_root = "/data1/lyf/Lab/VACE/deepcompressor_vace/examples/diffusion/datasets"
    dataset_name = "VACE-benchmark-real"
    pipeline_name = "Wan2.1-VACE-14B-lora-20steps"

    collect_dirpath = os.path.join(
        output_root,
        str(torch_dtype),
        pipeline_name,
        dataset_name,
        f"s{num_samples}",
    )
    print(f"Saving caches to {collect_dirpath}")

    samples = load_dataset(index_path, process_data_root, max_samples=num_samples)
    print(f"Loaded {len(samples)} samples from {index_path}")

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
        height=height,
        width=width,
    )
