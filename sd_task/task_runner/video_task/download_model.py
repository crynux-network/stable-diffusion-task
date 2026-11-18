import validators
from huggingface_hub import hf_hub_download

from sd_task.download_model import (
    best_guess_local_weight_name,
    check_and_download_hf_pipeline,
    check_and_download_model_by_name,
)
from sd_task.task_args.video_task import VideoTaskArgs
from sd_task.task_runner.image_task.download_model import (
    _download_diffusers_component,
    _resolve_external_model,
    download_component_file,
)
from sd_task.task_runner.image_task.errors import ModelNotDownloaded, TaskInvalid
from sd_task.task_runner.image_task.quantization.diffusers_component import (
    DIFFUSERS_COMPONENT_FORMAT,
)
from sd_task.task_runner.image_task.quantization.gguf import resolve_gguf_filename
from sd_task.task_runner.image_task.quantization.registry import (
    resolve_component_loader,
)


def check_and_prepare_models(task_args: VideoTaskArgs, **kwargs) -> dict[str, str]:
    task_args.base_model.name = check_and_download_hf_pipeline(
        task_args.base_model.name,
        revision=task_args.base_model.revision,
        **kwargs,
    )

    component_paths: dict[str, str] = {}
    for component_name, source in task_args.components.items():
        component_paths[component_name] = download_component_file(
            source, kwargs["hf_model_cache_dir"], kwargs.get("proxy")
        )

    for lora in task_args.loras:
        requested_weight = lora.weight_file_name
        lora.model, lora.weight_file_name = check_and_download_model_by_name(
            lora.model,
            None,
            [requested_weight] if requested_weight else [],
            not requested_weight,
            revision=lora.revision,
            **kwargs,
        )
    return component_paths


def resolve_models_from_cache(
    task_args: VideoTaskArgs,
    external_model_cache_dir: str,
    hf_model_cache_dir: str,
) -> dict[str, str]:
    component_paths: dict[str, str] = {}
    for component_name, source in task_args.components.items():
        format_name = resolve_component_loader(source)
        try:
            if format_name == "gguf":
                filename = resolve_gguf_filename(source)
                component_paths[component_name] = hf_hub_download(
                    repo_id=source.repo,
                    filename=filename,
                    revision=source.revision,
                    cache_dir=hf_model_cache_dir,
                    local_files_only=True,
                )
            elif format_name == DIFFUSERS_COMPONENT_FORMAT:
                component_paths[component_name] = _download_diffusers_component(
                    source, hf_model_cache_dir, local_files_only=True
                )
            else:
                raise TaskInvalid(f"Unsupported component format: {format_name}")
        except TaskInvalid:
            raise
        except Exception as exc:
            raise ModelNotDownloaded from exc

    for lora in task_args.loras:
        if validators.url(lora.model):
            lora.model, lora.weight_file_name = _resolve_external_model(
                lora.model, external_model_cache_dir
            )
        elif lora.weight_file_name == "":
            weight_file_name = best_guess_local_weight_name(
                lora.model,
                hf_model_cache_dir,
                revision=lora.revision,
            )
            if weight_file_name is None:
                raise ModelNotDownloaded from FileNotFoundError(
                    f"LoRA model {lora.model} is not in the local cache"
                )
            lora.weight_file_name = weight_file_name
        else:
            try:
                hf_hub_download(
                    repo_id=lora.model,
                    filename=lora.weight_file_name,
                    revision=lora.revision,
                    cache_dir=hf_model_cache_dir,
                    local_files_only=True,
                )
            except Exception as exc:
                raise ModelNotDownloaded from exc

    return component_paths
