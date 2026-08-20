# -*- coding: utf-8 -*-
"""Post-training quantization entry point for Wan2.1-VACE models.

This module registers the WanVideo pipeline factory and WanDiTStruct
into the DeepCompressor framework, then delegates to the standard
``ptq.main()`` so all YAML-based configurations work as expected.

Usage (from examples/diffusion/ directory):

  # Reference mode — inference only, save reference videos
  python -m deepcompressor.app.diffusion.ptq_vace \
      configs/model/wan2.1-vace-14b.yaml \
      --output-dirname reference \
      --skip-eval true

  # PTQ mode — quantize then (optionally) generate for comparison
  python -m deepcompressor.app.diffusion.ptq_vace \
      configs/model/wan2.1-vace-14b.yaml \
      configs/svdquant/nvfp4.yaml \
      --skip-eval true
"""

import copy
import glob
import json
import os
import pprint
import sys
import traceback

sys.path.insert(0, "/data1/lyf/Lab/DiffSynth-Studio")

import torch
from PIL import Image
from tqdm import tqdm

from deepcompressor.app.diffusion.nn.wan_struct import VaceWanDiTStruct, WanDiTStruct  # noqa: F401
from deepcompressor.app.diffusion.pipeline.config import DiffusionPipelineConfig
from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.app.diffusion.dataset.calib import DiffusionCalibCacheLoaderConfig
from deepcompressor.app.diffusion.dataset.calib_wan_loader import WanCalibCacheLoader
from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
from deepcompressor.app.diffusion.ptq import ptq
from deepcompressor.utils import tools
from deepcompressor.utils.common import hash_str_to_int

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video, VideoData


# ---------------------------------------------------------------------------
# Override build_loader so quantization uses WanCalibCacheLoader
# ---------------------------------------------------------------------------
_original_build_loader = DiffusionCalibCacheLoaderConfig.build_loader


def _wan_build_loader(self):
    return WanCalibCacheLoader(
        path=self.path,
        num_samples=self.num_samples,
        batch_size=self.batch_size,
    )


DiffusionCalibCacheLoaderConfig.build_loader = _wan_build_loader


def _build_wan_vace_pipeline(
    name: str,
    path: str,
    dtype: str | torch.dtype,
    device: str | torch.device,
    shift_activations: bool,
) -> WanVideoPipeline:
    model_base = path
    if name == "wan2.1-vace-1.3b":
        model_dir = "Wan-AI/Wan2.1-VACE-1.3B"
        lora_path = ""
    elif name == "wan2.1-vace-14b":
        model_dir = "Wan-AI/Wan2.1-VACE-14B"
        lora_path = "/data1/lyf/Lab/DiffSynth-Studio/lora/wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors"
    else:
        raise ValueError(f"Unsupported Wan VACE pipeline: {name}")
    model_configs = [
        ModelConfig(
            path=sorted(glob.glob(os.path.join(
                model_base, model_dir, "diffusion_pytorch_model*.safetensors"
            ))),
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
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    if lora_path:
        pipeline.load_lora(pipeline.dit, lora_path, alpha=1)
    return pipeline


DiffusionPipelineConfig.register_pipeline_factory(
    ("wan2.1-vace-14b", "wan2.1-vace-1.3b"), _build_wan_vace_pipeline
)


_orig_default_construct = DiffusionModelStruct._default_construct


def _extended_default_construct(
    module, /, parent=None, fname="", rname="", rkey="", idx=0, **kwargs
):
    if isinstance(module, WanVideoPipeline):
        module = module.dit
    from diffsynth.models.wan_video_dit import WanModel
    if isinstance(module, WanModel):
        return WanDiTStruct.construct(
            module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs
        )
    from diffsynth.models.wan_video_vace import VaceWanModel
    if isinstance(module, VaceWanModel):
        return VaceWanDiTStruct.construct(
            module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs
        )
    return _orig_default_construct(
        module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs
    )


DiffusionModelStruct._default_construct = staticmethod(_extended_default_construct)
DiffusionModelStruct.register_factory(WanVideoPipeline, _extended_default_construct, overwrite=True)


VACE_INDEX_PATH = "/data1/lyf/video_data/json/synthetic_fix.txt"
VACE_DATA_ROOT = "/data1/lyf/video_data/process_data"

VACE_PIPELINE_DEFAULTS = dict(
    num_frames=81,
    cfg_scale=5.0,
    negative_prompt=(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ),
    tiled=True,
    sigma_shift=5.0,
)


def _resolve_vace_quant_mode(config: DiffusionPtqRunConfig) -> int:
    """Resolve VACE quantization mode from env first, then YAML config.

    Modes:
      0: quantize Wan backbone and VACE branch
      1: quantize Wan backbone only
      2: quantize VACE branch only
    """

    value = os.environ.get("DEEPCOMPRESSOR_VACE_QUANT_MODE")
    if value is None:
        value = str(getattr(config, "vace_quant_mode", 0))
        # Backward compatibility with the older branch-only on/off switch.
        legacy_branch = os.environ.get("DEEPCOMPRESSOR_QUANT_VACE_BRANCH")
        if legacy_branch is not None and value == "0":
            if legacy_branch.lower() in ("0", "false", "no", "off"):
                value = "1"
    try:
        mode = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid VACE quantization mode: {value!r}") from exc
    if mode not in (0, 1, 2):
        raise ValueError("VACE quantization mode must be 0 (all), 1 (backbone), or 2 (branch)")
    return mode


def _build_vace_cache_config(config: DiffusionPtqRunConfig):
    """Create a stable global cache config for the VACE branch.

    DeepCompressor's normal PTQ cache is keyed by the pipeline/model name.
    The VACE side branch needs its own files so it does not overwrite the
    Wan backbone cache, but it should still reuse the same automatic cache
    discovery mechanism. We keep the same cache directories and add a
    ``.vace`` suffix to the cache filenames.
    """

    if config.cache is None or config.cache.path is None:
        return None

    vace_cache = copy.deepcopy(config.cache)

    def add_vace_suffix(path: str) -> str:
        if not path:
            return path
        stem, ext = os.path.splitext(path)
        return f"{stem}.vace{ext or '.pt'}"

    vace_cache.path = vace_cache.path.apply(add_vace_suffix)
    return vace_cache


def load_vace_dataset(index_path: str, data_root: str, max_samples: int = -1) -> list[dict]:
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
                    os.path.join(data_root, os.path.basename(p))
                    for p in item.get("src_ref_images", [])
                ],
            }
            samples.append(sample)
            if 0 < max_samples <= len(samples):
                break
    return samples


def build_vace_kwargs(sample: dict, height: int, width: int) -> dict:
    kwargs: dict = {}
    task = sample["task"]
    src_video_path = sample["src_video"]
    src_mask_path = sample["src_mask"]
    ref_images = sample["src_ref_images"]

    if task in ("MV2V", "V2V", "T2V"):
        kwargs["vace_video"] = VideoData(src_video_path, height=height, width=width)
        kwargs["vace_video_mask"] = VideoData(src_mask_path, height=height, width=width)
    elif task == "R2V":
        if ref_images:
            kwargs["vace_reference_image"] = Image.open(ref_images[0]).resize((width, height))

    return kwargs


def generate_vace(
    pipeline: WanVideoPipeline,
    output_dir: str,
    index_path: str = VACE_INDEX_PATH,
    data_root: str = VACE_DATA_ROOT,
    max_samples: int = -1,
    height: int = 480,
    width: int = 832,
    num_steps: int = 20,
    **extra_pipeline_kwargs,
) -> None:
    """Run VACE inference on the dataset and save output videos."""
    logger = tools.logging.getLogger(__name__)
    samples = load_vace_dataset(index_path, data_root, max_samples=max_samples)
    logger.info(f"Loaded {len(samples)} samples from {index_path}")

    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    base_kwargs = {
        **VACE_PIPELINE_DEFAULTS,
        "height": height,
        "width": width,
        "num_inference_steps": num_steps,
        **extra_pipeline_kwargs,
    }

    for sample in tqdm(samples, desc="VACE inference", dynamic_ncols=True):
        sample_id = sample["sample_id"]
        out_path = os.path.join(videos_dir, f"{sample_id}.mp4")
        if os.path.exists(out_path):
            continue

        prompt = sample["prompt"]
        seed = hash_str_to_int(sample_id)
        per_sample_kwargs = build_vace_kwargs(sample, height, width)

        result_video = pipeline(
            prompt=prompt,
            seed=seed,
            **base_kwargs,
            **per_sample_kwargs,
        )
        save_video(result_video, out_path, fps=15, quality=5)
        logger.info(f"Saved {out_path}")


def main_vace(config: DiffusionPtqRunConfig, logging_level: int = tools.logging.DEBUG):
    config.output.lock()
    config.dump(path=config.output.get_running_job_path("config.yaml"))
    tools.logging.setup(path=config.output.get_running_job_path("run.log"), level=logging_level)
    logger = tools.logging.getLogger(__name__)

    logger.info("=== Configurations ===")
    tools.logging.info(config.formatted_str(), logger=logger)
    logger.info("=== Dumped Configurations ===")
    tools.logging.info(pprint.pformat(config.dump(), indent=2, width=120), logger=logger)
    logger.info("=== Output Directory ===")
    logger.info(config.output.job_dirpath)

    logger.info("=== Building WanVideo pipeline ===")
    tools.logging.Formatter.indent_inc()
    pipeline = config.pipeline.build()
    assert isinstance(pipeline, WanVideoPipeline)

    if config.quant.is_enabled():
        tools.logging.Formatter.indent_dec()
        quant_mode = _resolve_vace_quant_mode(config)
        quant_backbone = quant_mode in (0, 1)
        quant_vace_branch = quant_mode in (0, 2)
        logger.info(
            "=== VACE quantization mode: %s (%s) ===",
            quant_mode,
            {0: "backbone+branch", 1: "backbone-only", 2: "branch-only"}[quant_mode],
        )

        save_dirpath = os.path.join(config.output.running_job_dirpath, "cache")
        if config.save_model:
            if config.save_model.lower() in ("false", "none", "null", "nil"):
                save_model = False
            elif config.save_model.lower() in ("true", "default"):
                save_dirpath = os.path.join(config.output.running_job_dirpath, "model")
                save_model = True
            else:
                save_dirpath, save_model = config.save_model, True
        else:
            save_model = False

        if quant_backbone:
            model = DiffusionModelStruct.construct(pipeline)
            logger.info("=== Quantizing WanModel ===")
            tools.logging.Formatter.indent_inc()
            model = ptq(
                model,
                config.quant,
                cache=config.cache,
                load_dirpath=config.load_from,
                save_dirpath=save_dirpath,
                copy_on_save=config.copy_on_save,
                save_model=save_model,
            )
            tools.logging.Formatter.indent_dec()
        else:
            logger.info("=== Skipping WanModel quantization (vace_quant_mode=2) ===")

        if getattr(pipeline, "vace", None) is not None and quant_vace_branch:
            logger.info("=== Quantizing VaceWanModel ===")
            tools.logging.Formatter.indent_inc()
            vace_model = VaceWanDiTStruct.construct(pipeline.vace, dit=pipeline.dit)
            vace_cache = _build_vace_cache_config(config)
            vace_save_dirpath = os.path.join(save_dirpath, "vace") if save_dirpath else ""
            vace_load_dirpath = os.path.join(config.load_from, "vace") if config.load_from else ""
            ptq(
                vace_model,
                config.quant,
                cache=vace_cache,
                load_dirpath=vace_load_dirpath,
                save_dirpath=vace_save_dirpath,
                copy_on_save=config.copy_on_save,
                save_model=save_model,
            )
            tools.logging.Formatter.indent_dec()
        elif getattr(pipeline, "vace", None) is not None:
            logger.info("=== Skipping VaceWanModel quantization (vace_quant_mode=1) ===")
    else:
        tools.logging.Formatter.indent_dec()
        logger.info("=== No quantization configured — reference mode ===")

    if not config.skip_gen:
        config.eval.gen_root = config.eval.gen_root.format(
            output=config.output.running_dirpath, job=config.output.running_job_dirname
        )
        gen_root = config.eval.gen_root or config.output.running_job_dirpath

        logger.info(f"=== Generating VACE videos → {gen_root} ===")
        tools.logging.Formatter.indent_inc()
        generate_vace(
            pipeline,
            output_dir=gen_root,
            index_path=VACE_INDEX_PATH,
            data_root=VACE_DATA_ROOT,
            max_samples=config.eval.num_samples,
            height=config.eval.height or 480,
            width=config.eval.width or 832,
            num_steps=config.eval.num_steps or 20,
        )
        tools.logging.Formatter.indent_dec()
    else:
        logger.info("=== Skipping generation (--skip-gen) ===")

    config.output.unlock()
    logger.info("=== Done ===")
    return pipeline


if __name__ == "__main__":
    config, _, unused_cfgs, unused_args, unknown_args = (
        DiffusionPtqRunConfig.get_parser().parse_known_args()
    )
    assert isinstance(config, DiffusionPtqRunConfig)
    if len(unused_cfgs) > 0:
        tools.logging.warning(f"Unused configurations: {unused_cfgs}")
    if unused_args is not None:
        tools.logging.warning(f"Unused arguments: {unused_args}")
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"
    try:
        main_vace(config, logging_level=tools.logging.DEBUG)
    except Exception as e:
        tools.logging.Formatter.indent_reset()
        tools.logging.error("=== Error ===")
        tools.logging.error(traceback.format_exc())
        tools.logging.shutdown()
        traceback.print_exc()
        config.output.unlock(error=True)
        raise e
