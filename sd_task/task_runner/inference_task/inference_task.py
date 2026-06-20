import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from diffusers import (AutoencoderKL, AutoPipelineForText2Image,
                       ControlNetModel, DiffusionPipeline,
                       UNet2DConditionModel)
from packaging.version import Version
from PIL import Image

from sd_task import utils, version
from sd_task.cache import ModelCache
from sd_task.config import Config, get_config
from sd_task.task_args.inference_task import (BaseModelArgs, ControlnetArgs,
                                              InferenceTaskArgs, RefinerArgs,
                                              TaskConfig)
from sd_task.log import log

from .controlnet import add_controlnet_pipeline_call_args
from .download_model import check_and_prepare_models
from .errors import (TaskVersionNotSupported, wrap_download_error,
                     wrap_execution_error)
from .key import generate_model_key
from .prompt import (add_prompt_pipeline_call_args,
                     add_prompt_refiner_sdxl_call_args)
from .scheduler import add_scheduler_pipeline_args


def get_pipeline_init_args(
    cache_dir: str, safety_checker: bool = True, torch_dtype: torch.dtype | None = None, variant: str | None = None
):
    init_args = {
        "torch_dtype": torch_dtype,
        "cache_dir": cache_dir,
        "local_files_only": True,
    }

    if variant is not None:
        init_args["variant"] = variant
        if variant == "fp16":
            init_args["torch_dtype"] = torch.float16

    if not safety_checker:
        init_args["safety_checker"] = None

    return init_args


def prepare_pipeline(cache_dir: str, args: InferenceTaskArgs):
    assert isinstance(args.base_model, BaseModelArgs)
    torch_dtype = None
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    pipeline_args = get_pipeline_init_args(
        cache_dir, args.task_config.safety_checker, torch_dtype, args.base_model.variant
    )
    acc_device = utils.get_accelerator()

    if args.controlnet is not None and args.controlnet.model != "":
        controlnet_model = None
        try:
            controlnet_model = ControlNetModel.from_pretrained(
                args.controlnet.model,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except EnvironmentError:
            pass

        if controlnet_model is None:
            controlnet_model = ControlNetModel.from_pretrained(
                args.controlnet.model,
                cache_dir=cache_dir,
                local_files_only=True,
            )

        pipeline_args["controlnet"] = controlnet_model.to(acc_device)

    if args.unet is not None and args.unet != "":
        unet_model = None
        try:
            unet_model = UNet2DConditionModel.from_pretrained(
                args.unet,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except EnvironmentError:
            pass

        if unet_model is None:
            unet_model = UNet2DConditionModel.from_pretrained(
                args.unet,
                cache_dir=cache_dir,
                local_files_only=True,
            )

        pipeline_args["unet"] = unet_model

    pipeline = AutoPipelineForText2Image.from_pretrained(
        args.base_model.name, **pipeline_args
    )

    add_scheduler_pipeline_args(pipeline, args.scheduler)

    if args.vae != "":
        vae_model = None
        try:
            vae_model = AutoencoderKL.from_pretrained(
                args.vae,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except EnvironmentError:
            pass

        if vae_model is None:
            vae_model = AutoencoderKL.from_pretrained(
                args.vae,
                cache_dir=cache_dir,
                local_files_only=True,
            )

        pipeline.vae = vae_model.to(acc_device)

    if args.lora is not None and args.lora.model != "":
        # raises ValueError if the lora model is not compatible with the base model
        pipeline.load_lora_weights(
            args.lora.model,
            weight_name=args.lora.weight_file_name,
            lora_scale=args.lora.weight,
            cache_dir=cache_dir,
            local_files_only=True,
        )

    if args.textual_inversion is not None and args.textual_inversion != "":
        pipeline.load_textual_inversion(
            args.textual_inversion, cache_dir=cache_dir, local_files_only=True
        )

    pipeline = pipeline.to(acc_device)

    # Refiner pipeline
    refiner_model = None

    if args.refiner is not None and args.refiner.model != "":
        refiner_init_args = get_pipeline_init_args(
            cache_dir, args.task_config.safety_checker, args.refiner.variant
        )
        refiner_init_args["tokenizer_2"] = pipeline.tokenizer_2
        refiner_init_args["text_encoder_2"] = pipeline.text_encoder_2
        refiner_init_args["vae"] = pipeline.vae
        refiner_model = DiffusionPipeline.from_pretrained(
            args.refiner.model, **refiner_init_args
        ).to(acc_device)

    return pipeline, refiner_model


def get_pipeline_call_args(
    pipeline,
    prompt: str,
    negative_prompt: str,
    task_config: TaskConfig,
    controlnet: ControlnetArgs | None = None,
    refiner: RefinerArgs | None = None,
) -> Dict[str, Any]:
    call_args: Dict[str, Any] = {
        "num_inference_steps": task_config.steps,
        "width": task_config.image_width,
        "height": task_config.image_height,
        "guidance_scale": task_config.cfg,
        # generator on CPU for reproducibility
        "generator": torch.Generator(device="cpu").manual_seed(task_config.seed),
    }

    add_prompt_pipeline_call_args(call_args, pipeline, prompt, negative_prompt)

    if controlnet is not None:
        add_controlnet_pipeline_call_args(
            call_args, controlnet, task_config.image_width, task_config.image_height
        )

    if refiner is not None:
        call_args["output_type"] = "latent"

        # denoising_end is not supported by StableDiffusionXLControlNetPipeline yet.
        if controlnet is None:
            call_args["denoising_end"] = refiner.denoising_cutoff

    return call_args


def run_inference_task(
    args: InferenceTaskArgs,
    config: Config | None = None,
    model_cache: ModelCache | None = None,
) -> List[Image.Image]:
    # Make sure the version of task is supported
    runner_version = Version(version())
    task_version = Version(args.version)

    if runner_version < task_version:
        raise TaskVersionNotSupported()

    if config is None:
        config = get_config()

    if config.deterministic:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)

    # Get GPU information
    gpu_info = utils.get_gpu_info()
    if gpu_info:
        log(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']}GB)")

    if config.deterministic and utils.get_accelerator() == "cuda":
        # Use deterministic CUDA backends for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"

    log("Task is started")

    torch.manual_seed(args.task_config.seed)
    random.seed(args.task_config.seed)
    np.random.seed(args.task_config.seed)

    model_key = generate_model_key(args)

    def model_loader():
        log("Check the model cache and download the models")

        with wrap_download_error():
            check_and_prepare_models(
                args,
                external_model_cache_dir=config.data_dir.models.external,
                hf_model_cache_dir=config.data_dir.models.huggingface,
                proxy=config.proxy,
            )

        log("All the required models are downloaded")

        with wrap_execution_error():
            pipeline, refiner = prepare_pipeline(
                cache_dir=config.data_dir.models.huggingface, args=args
            )
        log("The pipeline has been successfully loaded")
        return pipeline, refiner

    if model_cache is not None:
        pipeline, refiner = model_cache.load(model_key, model_loader)
    else:
        pipeline, refiner = model_loader()

    generated_images = []

    with wrap_execution_error():
        call_args = get_pipeline_call_args(
            pipeline,
            args.prompt,
            args.negative_prompt,
            args.task_config,
            args.controlnet,
            args.refiner,
        )

        refiner_call_args = {"generator": call_args["generator"]}

        if args.refiner is not None and refiner is not None:
            add_prompt_refiner_sdxl_call_args(
                refiner_call_args, refiner, args.prompt, args.negative_prompt
            )
            refiner_call_args["num_inference_steps"] = args.refiner.steps

            # denoising_end is not supported by StableDiffusionXLControlNetPipeline yet.
            if args.controlnet is None:
                refiner_call_args["denoising_start"] = args.refiner.denoising_cutoff

        log("The images generation is started")

        for _ in range(args.task_config.num_images):
            image = pipeline(**call_args)

            if refiner is not None:
                refiner_call_args["image"] = image.images
                image = refiner(**refiner_call_args)

            generated_images.append(image.images[0])
            del image

        del refiner_call_args
        del call_args

        log("The images generation is finished")

    if utils.get_accelerator() == "cuda":
        torch.cuda.empty_cache()

    return generated_images
