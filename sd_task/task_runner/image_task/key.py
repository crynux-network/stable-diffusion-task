import hashlib
import json
from typing import Any, Type

from diffusers import DiffusionPipeline

from sd_task.task_args.image_task import ImageTaskArgs


def generate_model_key(
    args: ImageTaskArgs, pipeline_class: Type[DiffusionPipeline]
) -> str:
    model_args: dict[str, Any] = {
        "pipeline_class": (
            f"{pipeline_class.__module__}.{pipeline_class.__qualname__}"
        ),
        "base_model": args.base_model.name,
        "dtype": args.dtype,
    }

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
