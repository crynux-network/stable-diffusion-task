import os

import validators
from huggingface_hub import hf_hub_download

from sd_task.download_model import (
    best_guess_local_weight_name,
    check_and_download_hf_pipeline,
    check_and_download_model_by_name,
    get_external_model_path,
    requests_proxy_session,
)
from sd_task.config import ProxyConfig
from sd_task.task_args.image_task import ComponentSource, ImageTaskArgs

from .errors import ModelNotDownloaded, TaskInvalid
from .quantization.gguf import resolve_gguf_filename
from .quantization.registry import resolve_component_loader


def download_component_file(
    source: ComponentSource, hf_model_cache_dir: str, proxy: ProxyConfig | None
) -> str:
    format_name = resolve_component_loader(source)
    if format_name != "gguf":
        raise TaskInvalid(f"Unsupported component format: {format_name}")
    filename = resolve_gguf_filename(source)
    with requests_proxy_session(proxy):
        return hf_hub_download(
            repo_id=source.repo,
            filename=filename,
            revision=source.revision,
            cache_dir=hf_model_cache_dir,
        )


def check_and_prepare_models(task_args: ImageTaskArgs, **kwargs) -> dict[str, str]:
    task_args.base_model.name = check_and_download_hf_pipeline(
        task_args.base_model.name, revision=task_args.base_model.revision, **kwargs
    )

    component_paths: dict[str, str] = {}
    for component_name, source in task_args.components.items():
        component_paths[component_name] = download_component_file(
            source, kwargs["hf_model_cache_dir"], kwargs.get("proxy")
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
    return component_paths


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
) -> dict[str, str]:
    component_paths: dict[str, str] = {}
    for component_name, source in task_args.components.items():
        format_name = resolve_component_loader(source)
        if format_name != "gguf":
            raise TaskInvalid(f"Unsupported component format: {format_name}")
        filename = resolve_gguf_filename(source)
        try:
            component_paths[component_name] = hf_hub_download(
                repo_id=source.repo,
                filename=filename,
                revision=source.revision,
                cache_dir=hf_model_cache_dir,
                local_files_only=True,
            )
        except Exception as exc:
            raise ModelNotDownloaded from exc

    if task_args.lora is None:
        return component_paths

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
    return component_paths
