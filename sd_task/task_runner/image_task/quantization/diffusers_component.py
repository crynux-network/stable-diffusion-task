from __future__ import annotations

from typing import Any

import torch

from sd_task.task_args.image_task import ComponentSource
from sd_task.task_runner.image_task.errors import TaskInvalid

from .component_class import component_class_from_pipeline_config
from .errors import UnsupportedComponentFormat

DIFFUSERS_COMPONENT_FORMAT = "diffusers"


def resolve_diffusers_subfolder(source: ComponentSource) -> str | None:
    if source.format != DIFFUSERS_COMPONENT_FORMAT:
        raise UnsupportedComponentFormat(source.format)
    subfolder = source.format_kwargs.get("subfolder")
    if subfolder is None:
        return None
    if not isinstance(subfolder, str) or not subfolder:
        raise TaskInvalid(
            "Diffusers component format_kwargs.subfolder must be a non-empty string"
        )
    return subfolder


def load_diffusers_component(
    *,
    pipeline_config: dict[str, Any],
    component_name: str,
    source: ComponentSource,
    cache_dir: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
    component_path: str,
) -> Any:
    if source.format != DIFFUSERS_COMPONENT_FORMAT:
        raise UnsupportedComponentFormat(source.format)

    component_cls = component_class_from_pipeline_config(
        pipeline_config, component_name
    )
    if not hasattr(component_cls, "from_pretrained"):
        raise TaskInvalid(
            f"Component class for {component_name} does not support from_pretrained"
        )

    load_args: dict[str, Any] = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "torch_dtype": torch_dtype,
    }
    subfolder = resolve_diffusers_subfolder(source)
    if subfolder is not None:
        load_args["subfolder"] = subfolder

    return component_cls.from_pretrained(component_path, **load_args)
