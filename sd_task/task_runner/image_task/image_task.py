import os
import random

import numpy as np
import torch
from packaging.version import Version

from sd_task import utils, version
from sd_task.cache import ModelCache
from sd_task.config import Config, get_config
from sd_task.log import log
from sd_task.task_args.image_task import ImageTaskArgs

from .download_model import check_and_prepare_models, resolve_models_from_cache
from .errors import (
    TaskVersionNotSupported,
    wrap_download_error,
    wrap_execution_error,
    wrap_model_load_error,
)
from .key import generate_model_key
from .pipeline import (
    build_pipeline_call_args,
    load_lora,
    load_pipeline,
    move_pipeline_to_accelerator,
    resolve_pipeline_class,
)
from .result import ImageTaskResult, resolve_execution_dtype


def _resolve_torch_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return None


def _configure_determinism(config: Config, seed: int) -> None:
    if config.deterministic:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)

    if config.deterministic and utils.get_accelerator() == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _prepare_models(args: ImageTaskArgs, config: Config) -> None:
    if config.local_files_only:
        log("Resolve the models from the local cache only")
        resolve_models_from_cache(
            args,
            external_model_cache_dir=config.data_dir.models.external,
            hf_model_cache_dir=config.data_dir.models.huggingface,
        )
        return

    log("Check the model cache and download the models")
    with wrap_download_error():
        check_and_prepare_models(
            args,
            external_model_cache_dir=config.data_dir.models.external,
            hf_model_cache_dir=config.data_dir.models.huggingface,
            proxy=config.proxy,
        )
    log("All the required models are downloaded")


def run_image_task(
    args: ImageTaskArgs,
    config: Config | None = None,
    model_cache: ModelCache | None = None,
) -> ImageTaskResult:
    if Version(version()) < Version(args.version):
        raise TaskVersionNotSupported()

    if config is None:
        config = get_config()

    _configure_determinism(config, args.task_config.seed)

    gpu_info = utils.get_gpu_info()
    if gpu_info:
        log(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']}GB)")
    log("Task is started")

    _prepare_models(args, config)

    with wrap_model_load_error(config.local_files_only):
        pipeline_class = resolve_pipeline_class(
            args.base_model.name,
            cache_dir=config.data_dir.models.huggingface,
            local_files_only=True,
            has_images=bool(args.images),
        )

    model_key = generate_model_key(args, pipeline_class)

    def model_loader():
        with wrap_model_load_error(config.local_files_only):
            pipeline = load_pipeline(
                pipeline_class,
                args.base_model.name,
                cache_dir=config.data_dir.models.huggingface,
                torch_dtype=_resolve_torch_dtype(args.dtype),
                local_files_only=True,
            )
            if args.lora is not None:
                load_lora(
                    pipeline,
                    model=args.lora.model,
                    weight_file_name=args.lora.weight_file_name,
                    weight=args.lora.weight,
                    cache_dir=config.data_dir.models.huggingface,
                    local_files_only=True,
                )
            pipeline = move_pipeline_to_accelerator(pipeline)
        log("The pipeline has been successfully loaded")
        return pipeline

    if model_cache is not None:
        pipeline = model_cache.load(model_key, model_loader)
    else:
        pipeline = model_loader()

    execution_dtype = resolve_execution_dtype(pipeline)
    call_args = build_pipeline_call_args(pipeline, args)

    with wrap_execution_error():
        log("The images generation is started")
        result = pipeline(**call_args)
        generated_images = list(result.images)
        if len(generated_images) != args.task_config.num_images_per_prompt:
            raise RuntimeError("The pipeline returned an unexpected number of images")
        log("The images generation is finished")

    if utils.get_accelerator() == "cuda":
        torch.cuda.empty_cache()

    return ImageTaskResult(generated_images, execution_dtype)
