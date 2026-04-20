# -*- coding: utf-8 -*-
"""Post-training quantization entry point for Wan2.1 T2V (1.3B / 14B).

Registers WanVideo pipeline factories and ``WanDiTStruct``, then delegates to
``ptq.main()`` for YAML-driven PTQ.

Usage (from examples/diffusion/ directory):

  # Wan2.1-T2V-1.3B — reference / PTQ
  python -m deepcompressor.app.diffusion.ptq_wan_t2v \\
      configs/model/wan2.1-t2v-1.3b.yaml \\
      --output-dirname reference --skip-eval true

  python -m deepcompressor.app.diffusion.ptq_wan_t2v \\
      configs/model/wan2.1-t2v-1.3b.yaml configs/svdquant/nvfp4.yaml \\
      --skip-eval true

  # Wan2.1-T2V-14B — use ``configs/model/wan2.1-t2v-14b.yaml`` (pipeline.name: wan2.1-t2v-14b)
"""

import glob
import json
import os
import pprint
import sys
import traceback

sys.path.insert(0, "/data1/lyf/Lab/DiffSynth-Studio")

import torch
from tqdm import tqdm

from deepcompressor.app.diffusion.nn.wan_struct import WanDiTStruct  # noqa: F401
from deepcompressor.app.diffusion.pipeline.config import DiffusionPipelineConfig
from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.app.diffusion.dataset.calib import DiffusionCalibCacheLoaderConfig
from deepcompressor.app.diffusion.dataset.calib_wan_loader import WanCalibCacheLoader
from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
from deepcompressor.app.diffusion.ptq import ptq
from deepcompressor.utils import tools
from deepcompressor.utils.common import hash_str_to_int

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


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


# ---------------------------------------------------------------------------
# Pipeline factory for Wan2.1-T2V-1.3B
# ---------------------------------------------------------------------------
def _build_wan_t2v_1_3b_pipeline(
    name: str,
    path: str,
    dtype: str | torch.dtype,
    device: str | torch.device,
    shift_activations: bool,
) -> WanVideoPipeline:
    model_base = path
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
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    return pipeline


DiffusionPipelineConfig.register_pipeline_factory(
    "wan2.1-t2v-1.3b", _build_wan_t2v_1_3b_pipeline
)


# ---------------------------------------------------------------------------
# Pipeline factory for Wan2.1-T2V-14B
# ---------------------------------------------------------------------------
def _build_wan_t2v_14b_pipeline(
    name: str,
    path: str,
    dtype: str | torch.dtype,
    device: str | torch.device,
    shift_activations: bool,
) -> WanVideoPipeline:
    model_base = path
    dit_glob = os.path.join(
        model_base,
        "Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model*.safetensors",
    )
    dit_paths = sorted(glob.glob(dit_glob))
    if not dit_paths:
        raise FileNotFoundError(
            f"Wan2.1-T2V-14B: no DiT weights matching {dit_glob!r}. "
            "Place diffusion_pytorch_model*.safetensors under that folder."
        )
    model_configs = [
        ModelConfig(
            path=dit_paths if len(dit_paths) > 1 else dit_paths[0],
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
    
    return pipeline


DiffusionPipelineConfig.register_pipeline_factory(
    "wan2.1-t2v-14b", _build_wan_t2v_14b_pipeline
)


# ---------------------------------------------------------------------------
# Extend DiffusionModelStruct to handle WanModel / WanVideoPipeline
# ---------------------------------------------------------------------------
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
    return _orig_default_construct(
        module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs
    )


DiffusionModelStruct._default_construct = staticmethod(_extended_default_construct)
DiffusionModelStruct.register_factory(
    WanVideoPipeline, _extended_default_construct, overwrite=True
)


# ---------------------------------------------------------------------------
# T2V evaluation dataset & generation
# ---------------------------------------------------------------------------
# 默认使用 0001 分卷 caption（与 0000 目录校准数据错开）。可用环境变量覆盖：
#   WAN_T2V_CAPTION_JSON=/path/to.json WAN_T2V_VIDEO_ROOT=/path/to/source/source
T2V_CAPTION_JSON = os.environ.get(
    "WAN_T2V_CAPTION_JSON",
    "/data1/lyf/Lab/Video_dataset/Ditto-1M/source_video_captions/"
    "source_video_captions_0001.json",
)
T2V_VIDEO_ROOT = os.environ.get(
    "WAN_T2V_VIDEO_ROOT",
    "/data1/lyf/Lab/Video_dataset/Ditto-1M/videos/source/source",
)

T2V_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def load_t2v_dataset(
    caption_json_path: str,
    video_root: str,
    max_samples: int = -1,
) -> list[dict]:
    """
    available_dirs = set(os.listdir(video_root))
    with open(caption_json_path, "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    samples = []
    """
    _ = (caption_json_path, video_root, max_samples)
    samples = [
        {
            "sample_id": "000001.mp4",
            "prompt": (
                "The video showcases an expansive aerial view of a snow-covered urban landscape during "
                "what appears to be either dawn or dusk, as indicated by the soft, muted light in the "
                "sky. The scene is dominated by a sprawling cityscape with numerous buildings, many of "
                "which have flat roofs blanketed in snow. The architecture is predominantly low-rise, "
                "with some taller structures scattered throughout, suggesting a mix of residential and "
                "industrial areas.\n\n"
                "In the foreground, there are clusters of smaller buildings, possibly warehouses or "
                "workshops, interspersed with open spaces that appear to be parking lots or storage "
                "areas. The middle ground features a more densely packed urban area with rows of "
                "similar-looking buildings, likely residential complexes. In the background, a gently "
                "sloping hill covered in snow rises, adding depth to the scene. The sky is overcast "
                "with a gradient of colors transitioning from a pale orange near the horizon to a "
                "grayish-blue higher up, indicating the time of day.\n\n"
                "The camera maintains a steady, wide-angle perspective throughout the sequence, "
                "capturing the vastness of the snowy landscape without any noticeable movement such as "
                "panning or zooming. This stationary viewpoint allows for a comprehensive overview of "
                "the city's layout and the surrounding natural environment, emphasizing the stark "
                "contrast between the built structures and the untouched snow-covered terrain. The "
                "overall atmosphere is serene and cold, underscored by the pervasive whiteness of the "
                "snow and the subdued lighting."
            ),
        }
    ]
    """
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
        })
        if 0 < max_samples <= len(samples):
            break
    """  
    return samples


def generate_t2v(
    pipeline: WanVideoPipeline,
    output_dir: str,
    caption_json_path: str = T2V_CAPTION_JSON,
    video_root: str = T2V_VIDEO_ROOT,
    max_samples: int = -1,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_steps: int = 50,
    cfg_scale: float = 5.0,
) -> None:
    logger = tools.logging.getLogger(__name__)
    samples = load_t2v_dataset(caption_json_path, video_root, max_samples=max_samples)
    logger.info(f"Loaded {len(samples)} T2V samples for evaluation")

    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    for idx, sample in enumerate(
        tqdm(samples, desc="T2V evaluation", dynamic_ncols=True), start=1
    ):
        seq_id = f"{idx:06d}"
        out_path = os.path.join(videos_dir, f"{seq_id}.mp4")
        if os.path.exists(out_path):
            continue

        seed = hash_str_to_int(seq_id)
        result_video = pipeline(
            prompt=sample["prompt"],
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            cfg_scale=cfg_scale,
            negative_prompt=T2V_NEGATIVE_PROMPT,
            tiled=True,
            sigma_shift=5.0,
        )
        save_video(result_video, out_path, fps=15, quality=5)
        logger.info(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main_t2v(config: DiffusionPtqRunConfig, logging_level: int = tools.logging.DEBUG):
    config.output.lock()
    config.dump(path=config.output.get_running_job_path("config.yaml"))
    tools.logging.setup(
        path=config.output.get_running_job_path("run.log"), level=logging_level
    )
    logger = tools.logging.getLogger(__name__)

    logger.info("=== Configurations ===")
    tools.logging.info(config.formatted_str(), logger=logger)
    logger.info("=== Dumped Configurations ===")
    tools.logging.info(
        pprint.pformat(config.dump(), indent=2, width=120), logger=logger
    )
    logger.info("=== Output Directory ===")
    logger.info(config.output.job_dirpath)

    logger.info(f"=== Building Wan T2V pipeline ({config.pipeline.name}) ===")
    tools.logging.Formatter.indent_inc()
    pipeline = config.pipeline.build()
    assert isinstance(pipeline, WanVideoPipeline)

    if config.quant.is_enabled():
        model = DiffusionModelStruct.construct(pipeline)
        tools.logging.Formatter.indent_dec()

        save_dirpath = os.path.join(config.output.running_job_dirpath, "cache")
        if config.save_model:
            if config.save_model.lower() in ("false", "none", "null", "nil"):
                save_model = False
            elif config.save_model.lower() in ("true", "default"):
                save_dirpath = os.path.join(
                    config.output.running_job_dirpath, "model"
                )
                save_model = True
            else:
                save_dirpath, save_model = config.save_model, True
        else:
            save_model = False

        logger.info(f"=== Quantizing WanModel ({config.pipeline.name}) ===")
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
        tools.logging.Formatter.indent_dec()
        logger.info("=== No quantization configured -- reference mode ===")

    if not config.skip_gen:
        config.eval.gen_root = config.eval.gen_root.format(
            output=config.output.running_dirpath,
            job=config.output.running_job_dirname,
        )
        gen_root = config.eval.gen_root or config.output.running_job_dirpath

        logger.info(f"=== Generating T2V videos -> {gen_root} ===")
        tools.logging.Formatter.indent_inc()
        if config.pipeline.name == "wan2.1-t2v-14b":
            pipeline.load_lora(pipeline.dit, "/data1/lyf/Lab/DiffSynth-Studio/lora/wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors", alpha=1)
        generate_t2v(
            pipeline,
            output_dir=gen_root,
            caption_json_path=T2V_CAPTION_JSON,
            video_root=T2V_VIDEO_ROOT,
            max_samples=config.eval.num_samples,
            height=config.eval.height or 480,
            width=config.eval.width or 832,
            num_steps=config.eval.num_steps or 40,
            cfg_scale=config.eval.guidance_scale or 5.0,
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
        main_t2v(config, logging_level=tools.logging.DEBUG)
    except Exception as e:
        tools.logging.Formatter.indent_reset()
        tools.logging.error("=== Error ===")
        tools.logging.error(traceback.format_exc())
        tools.logging.shutdown()
        traceback.print_exc()
        config.output.unlock(error=True)
        raise e
