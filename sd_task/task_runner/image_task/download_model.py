import os

import validators

from sd_task.download_model import (
    best_guess_local_weight_name,
    check_and_download_hf_pipeline,
    check_and_download_model_by_name,
    get_external_model_path,
)
from sd_task.task_args.image_task import ImageTaskArgs

from .errors import ModelNotDownloaded


def check_and_prepare_models(task_args: ImageTaskArgs, **kwargs) -> None:
    task_args.base_model.name = check_and_download_hf_pipeline(
        task_args.base_model.name, **kwargs
    )

    if task_args.lora is not None:
        requested_weight = task_args.lora.weight_file_name
        task_args.lora.model, task_args.lora.weight_file_name = (
            check_and_download_model_by_name(
                task_args.lora.model,
                None,
                [requested_weight] if requested_weight else [],
                not requested_weight,
                **kwargs,
            )
        )


def _resolve_external_model(
    model_name: str, external_model_cache_dir: str
) -> tuple[str, str]:
    model_file, weight_file_name = get_external_model_path(
        model_name, external_model_cache_dir
    )
    if not os.path.isfile(model_file):
        raise ModelNotDownloaded from FileNotFoundError(
            f"External model {model_name} is not in the local cache"
        )
    return model_file, weight_file_name


def resolve_models_from_cache(
    task_args: ImageTaskArgs,
    external_model_cache_dir: str,
    hf_model_cache_dir: str,
) -> None:
    if task_args.lora is None:
        return

    if validators.url(task_args.lora.model):
        task_args.lora.model, task_args.lora.weight_file_name = _resolve_external_model(
            task_args.lora.model, external_model_cache_dir
        )
    elif task_args.lora.weight_file_name == "":
        weight_file_name = best_guess_local_weight_name(
            task_args.lora.model, hf_model_cache_dir
        )
        if weight_file_name is None:
            raise ModelNotDownloaded from FileNotFoundError(
                f"Lora model {task_args.lora.model} is not in the local cache"
            )
        task_args.lora.weight_file_name = weight_file_name
