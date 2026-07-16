import os

import validators
from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME

from sd_task.download_model import (best_guess_local_weight_name,
                                    check_and_download_hf_pipeline,
                                    check_and_download_model_by_name,
                                    get_external_model_path)
from sd_task.task_args.inference_task import (BaseModelArgs, InferenceTaskArgs)

from .errors import ModelNotDownloaded


def check_and_prepare_models(task_args: InferenceTaskArgs, **kwargs):

    assert isinstance(task_args.base_model, BaseModelArgs)
    task_args.base_model.name = check_and_download_hf_pipeline(
        task_args.base_model.name, task_args.base_model.variant, **kwargs
    )

    if task_args.refiner is not None and task_args.refiner.model != "":
        task_args.refiner.model = check_and_download_hf_pipeline(
            task_args.refiner.model, task_args.refiner.variant, **kwargs
        )

    if task_args.unet is not None and task_args.unet != "":
        task_args.unet, _ = check_and_download_model_by_name(
            task_args.unet,
            UNet2DConditionModel.load_config,
            [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
            False,
            **kwargs,
        )

    if task_args.vae != "":
        task_args.vae, _ = check_and_download_model_by_name(
            task_args.vae,
            AutoencoderKL.load_config,
            [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
            False,
            **kwargs,
        )

    if task_args.controlnet is not None:
        task_args.controlnet.model, _ = check_and_download_model_by_name(
            task_args.controlnet.model,
            ControlNetModel.load_config,
            [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
            False,
            variant=task_args.controlnet.variant,
            **kwargs,
        )

    if task_args.lora is not None:
        task_args.lora.model, task_args.lora.weight_file_name = (
            check_and_download_model_by_name(
                task_args.lora.model, None, [], True, **kwargs
            )
        )

    if task_args.textual_inversion != "":
        task_args.textual_inversion, _ = check_and_download_model_by_name(
            task_args.textual_inversion, None, [], True, **kwargs
        )


def _resolve_external_model(model_name: str, external_model_cache_dir: str) -> tuple[str, str]:
    model_file, weight_file_name = get_external_model_path(
        model_name, external_model_cache_dir
    )
    if not os.path.isfile(model_file):
        raise ModelNotDownloaded from FileNotFoundError(
            f"External model {model_name} is not in the local cache"
        )
    return model_file, weight_file_name


# Resolves external (URL) model names to local cache paths without any network
# access, for local_files_only mode. Hugging Face model names are left
# untouched: the pipeline loads them with local_files_only=True and a cache
# miss surfaces as ModelNotDownloaded at load time.
def resolve_models_from_cache(
    task_args: InferenceTaskArgs,
    external_model_cache_dir: str,
    hf_model_cache_dir: str,
):
    if task_args.unet is not None and task_args.unet != "" and validators.url(task_args.unet):
        task_args.unet, _ = _resolve_external_model(
            task_args.unet, external_model_cache_dir
        )

    if task_args.vae != "" and validators.url(task_args.vae):
        task_args.vae, _ = _resolve_external_model(
            task_args.vae, external_model_cache_dir
        )

    if task_args.controlnet is not None and validators.url(task_args.controlnet.model):
        task_args.controlnet.model, _ = _resolve_external_model(
            task_args.controlnet.model, external_model_cache_dir
        )

    if task_args.lora is not None and task_args.lora.model != "":
        if validators.url(task_args.lora.model):
            task_args.lora.model, task_args.lora.weight_file_name = (
                _resolve_external_model(
                    task_args.lora.model, external_model_cache_dir
                )
            )
        elif task_args.lora.weight_file_name == "":
            # Offline lora loading requires an explicit weight file name;
            # guess it from the local snapshot instead of the hub API
            weight_file_name = best_guess_local_weight_name(
                task_args.lora.model, hf_model_cache_dir
            )
            if weight_file_name is None:
                raise ModelNotDownloaded from FileNotFoundError(
                    f"Lora model {task_args.lora.model} is not in the local cache"
                )
            task_args.lora.weight_file_name = weight_file_name

    if task_args.textual_inversion != "" and validators.url(task_args.textual_inversion):
        task_args.textual_inversion, _ = _resolve_external_model(
            task_args.textual_inversion, external_model_cache_dir
        )
