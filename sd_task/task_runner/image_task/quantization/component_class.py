from __future__ import annotations

from typing import Any, Type

from diffusers import DiffusionPipeline

from sd_task.task_runner.image_task.errors import TaskInvalid


def resolve_component_class(library_name: str, class_name: str) -> Type[Any]:
    if library_name == "diffusers":
        module = __import__("diffusers")
    elif library_name == "transformers":
        module = __import__("transformers")
    else:
        raise TaskInvalid(f"Unsupported component library: {library_name}")
    component_cls = getattr(module, class_name, None)
    if component_cls is None:
        raise TaskInvalid(f"Component class {class_name} is unavailable")
    return component_cls


def pipeline_config_from_base(
    model_name: str,
    cache_dir: str,
    local_files_only: bool,
    revision: str | None = None,
) -> dict[str, Any]:
    load_args: dict[str, Any] = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        load_args["revision"] = revision
    config = DiffusionPipeline.load_config(model_name, **load_args)
    if not isinstance(config, dict):
        raise TaskInvalid("Pipeline config is invalid")
    return config


def component_class_from_pipeline_config(
    pipeline_config: dict[str, Any], component_name: str
) -> Type[Any]:
    entry = pipeline_config.get(component_name)
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise TaskInvalid(
            f"Pipeline model_index.json does not declare {component_name}"
        )
    library_name, class_name = entry
    if not isinstance(library_name, str) or not isinstance(class_name, str):
        raise TaskInvalid(
            f"Pipeline {component_name} class declaration is invalid"
        )
    return resolve_component_class(library_name, class_name)
