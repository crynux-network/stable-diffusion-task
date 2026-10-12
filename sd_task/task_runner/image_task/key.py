import hashlib
import json
from typing import Any, Type

from diffusers import DiffusionPipeline

from sd_task.task_args.image_task import ImageTaskArgs


def generate_model_key(
    args: ImageTaskArgs,
    pipeline_class: Type[DiffusionPipeline],
    *,
    effective_quantization: dict[str, Any] | None = None,
    component_revisions: dict[str, str] | None = None,
) -> str:
    model_args: dict[str, Any] = {
        "pipeline_class": (
            f"{pipeline_class.__module__}.{pipeline_class.__qualname__}"
        ),
        "base_model": {
            "name": args.base_model.name,
            "revision": args.base_model.revision,
        },
        "dtype": args.dtype,
    }

    if args.components:
        components: dict[str, Any] = {}
        for name, source in sorted(args.components.items()):
            entry = source.model_dump(mode="json")
            if component_revisions and name in component_revisions:
                entry["resolved_revision"] = component_revisions[name]
            components[name] = entry
        model_args["components"] = components

    if args.quantization_config is not None:
        model_args["quantization_config"] = (
            args.quantization_config.model_dump(mode="json")
        )
    elif effective_quantization is not None:
        model_args["quantization_config"] = effective_quantization

    if args.lora is not None:
        model_args["lora"] = {
            "model": args.lora.model,
            "weight_file_name": args.lora.weight_file_name,
            "weight": args.lora.weight,
        }

    model_args_str = json.dumps(
        model_args, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(model_args_str.encode("utf-8")).hexdigest()
