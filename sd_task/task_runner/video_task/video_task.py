from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from diffusers import ModularPipeline
from packaging.version import Version

from sd_task import utils, version
from sd_task.cache import ModelCache
from sd_task.config import Config, get_config
from sd_task.log import log
from sd_task.task_args.video_task import VideoTaskArgs, derive_video_input_mode
from sd_task.task_runner.image_task.errors import (
    TaskVersionNotSupported,
    wrap_download_error,
    wrap_execution_error,
    wrap_model_load_error,
)

from .download_model import check_and_prepare_models, resolve_models_from_cache
from .dtype import resolve_video_dtype
from .encode import encode_canonical_mp4
from .input_adapter import build_video_call_args, resolve_effective_fps
from .key import generate_video_model_key
from .pipeline import (
    configure_vae_and_accelerator,
    load_video_loras,
    load_video_pipeline,
    resolve_video_pipeline_spec,
)
from .result import VideoTaskResult, resolve_video_execution_dtype


def _configure_video_determinism(config: Config, seed: int) -> None:
    if config.deterministic:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

    if config.deterministic and utils.get_accelerator() == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _prepare_video_models(args: VideoTaskArgs, config: Config) -> dict[str, str]:
    if config.local_files_only:
        log("Resolve the video models from the local cache only")
        return resolve_models_from_cache(
            args,
            external_model_cache_dir=config.data_dir.models.external,
            hf_model_cache_dir=config.data_dir.models.huggingface,
        )

    log("Check the model cache and download the video models")
    with wrap_download_error():
        component_paths = check_and_prepare_models(
            args,
            external_model_cache_dir=config.data_dir.models.external,
            hf_model_cache_dir=config.data_dir.models.huggingface,
            proxy=config.proxy,
        )
    log("All the required video models are downloaded")
    return component_paths


def _extract_pipeline_outputs(
    pipeline: Any, raw_output: Any
) -> tuple[Any, Any, int | None]:
    frames: Any = None
    audio: Any = None
    sample_rate: int | None = None

    if isinstance(pipeline, ModularPipeline):
        values = getattr(raw_output, "values", None)
        if isinstance(values, dict):
            frames = values.get("videos", values.get("frames"))
            audio = values.get("audio")
            sample_rate = values.get("sampling_rate", values.get("audio_sampling_rate"))
    if frames is None:
        if hasattr(raw_output, "frames"):
            frames = raw_output.frames
            audio = getattr(raw_output, "audio", None)
            sample_rate = getattr(
                raw_output,
                "sampling_rate",
                getattr(raw_output, "audio_sampling_rate", None),
            )
        elif hasattr(raw_output, "videos"):
            frames = raw_output.videos
            audio = getattr(raw_output, "audio", None)
            sample_rate = getattr(raw_output, "sampling_rate", None)
        elif isinstance(raw_output, tuple) and len(raw_output) >= 1:
            frames = raw_output[0]
            if len(raw_output) >= 2:
                audio = raw_output[1]

    if frames is None:
        raise RuntimeError("Video pipeline output did not contain generated frames")

    if audio is not None and sample_rate is None:
        vocoder = getattr(pipeline, "vocoder", None)
        vocoder_cfg = getattr(vocoder, "config", None)
        sample_rate = getattr(vocoder_cfg, "output_sampling_rate", None)
        if sample_rate is None:
            sample_rate = getattr(pipeline, "audio_sampling_rate", None)
        if sample_rate is not None:
            sample_rate = int(sample_rate)

    return frames, audio, sample_rate


def run_video_task(
    args: VideoTaskArgs,
    *,
    output_dir: str,
    config: Config | None = None,
    model_cache: ModelCache | None = None,
) -> VideoTaskResult:
    from sd_task.task_runner.image_task.quantization.registry import (
        resolve_image_quantization,
    )

    if Version(version()) < Version(args.version):
        raise TaskVersionNotSupported()

    if config is None:
        config = get_config()

    _configure_video_determinism(config, args.task_config.seed)

    gpu_info = utils.get_gpu_info()
    if gpu_info:
        log(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']}GB)")
    log("Video task is started")

    args = args.model_copy(deep=True)
    component_paths = _prepare_video_models(args, config)
    input_mode = derive_video_input_mode(args)

    with wrap_model_load_error(config.local_files_only):
        spec = resolve_video_pipeline_spec(
            args,
            cache_dir=config.data_dir.models.huggingface,
            local_files_only=True,
        )
        resolved_dtype_name, torch_dtype = resolve_video_dtype(
            args.dtype,
            base_model_name=args.base_model.name,
            base_model_revision=args.base_model.revision,
            input_mode=input_mode,
            cache_dir=config.data_dir.models.huggingface,
            local_files_only=True,
        )

    args = args.model_copy(update={"dtype": resolved_dtype_name})
    quantization = resolve_image_quantization(args)
    model_key = generate_video_model_key(
        args,
        spec,
        effective_quantization=(
            None if quantization is None else quantization["effective_config"]
        ),
    )

    def model_loader():
        with wrap_model_load_error(config.local_files_only):
            loaded = load_video_pipeline(
                spec,
                args,
                cache_dir=config.data_dir.models.huggingface,
                torch_dtype=torch_dtype,
                local_files_only=True,
                component_paths=component_paths,
                quantization=quantization,
            )
            if args.loras:
                load_video_loras(
                    loaded,
                    args.loras,
                    cache_dir=config.data_dir.models.huggingface,
                    local_files_only=True,
                )
            loaded = configure_vae_and_accelerator(loaded)
        log("The video pipeline has been successfully loaded")
        return loaded

    if model_cache is not None:
        pipeline = model_cache.load(model_key, model_loader)
    else:
        pipeline = model_loader()

    execution_dtype = resolve_video_execution_dtype(pipeline, quantization=quantization)
    effective_fps = resolve_effective_fps(pipeline, args)
    call_args, guider_restore = build_video_call_args(pipeline, args)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "0.mp4")

    try:
        with wrap_execution_error():
            log("The video generation is started")
            raw_output = pipeline(**call_args)
            frames, audio, audio_sample_rate = _extract_pipeline_outputs(
                pipeline, raw_output
            )
            num_frames, width, height, has_audio = encode_canonical_mp4(
                frames,
                fps=effective_fps,
                output_path=output_path,
                audio=audio,
                audio_sample_rate=audio_sample_rate,
            )
            if num_frames != args.task_config.num_frames:
                raise RuntimeError(
                    f"The video pipeline returned {num_frames} frames, "
                    f"expected {args.task_config.num_frames}"
                )
            log("The video generation is finished")
    finally:
        if guider_restore and isinstance(pipeline, ModularPipeline):
            pipeline.update_components(**guider_restore)

    if utils.get_accelerator() == "cuda":
        torch.cuda.empty_cache()

    return VideoTaskResult(
        output_path=output_path,
        num_frames=num_frames,
        width=width,
        height=height,
        fps=effective_fps,
        has_audio=has_audio,
        execution_dtype=execution_dtype,
        execution_quantization=(
            None if quantization is None else quantization["effective_config"]
        ),
    )
