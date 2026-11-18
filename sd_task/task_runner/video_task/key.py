from __future__ import annotations

import hashlib
import json
from typing import Any

from sd_task.task_args.video_task import VideoTaskArgs, derive_video_input_mode

from .pipeline import ResolvedVideoPipelineSpec


def generate_video_model_key(
    args: VideoTaskArgs,
    spec: ResolvedVideoPipelineSpec,
    *,
    effective_quantization: dict[str, Any] | None = None,
) -> str:
    model_args: dict[str, Any] = {
        "task_kind": "video",
        "input_mode": derive_video_input_mode(args),
        "pipeline_kind": spec.kind,
        "pipeline_class": (
            f"{spec.pipeline_class.__module__}.{spec.pipeline_class.__qualname__}"
        ),
        "workflow": spec.workflow,
        "base_model": {
            "name": args.base_model.name,
            "revision": args.base_model.revision,
        },
        "dtype": args.dtype,
    }

    if args.components:
        components: dict[str, Any] = {}
        for name, source in sorted(args.components.items()):
            components[name] = source.model_dump(mode="json")
        model_args["components"] = components

    if args.quantization_config is not None:
        model_args["quantization_config"] = (
            args.quantization_config.model_dump(mode="json")
        )
    elif effective_quantization is not None:
        model_args["quantization_config"] = effective_quantization

    if args.loras:
        model_args["loras"] = [
            {
                "model": lora.model,
                "revision": lora.revision,
                "weight_file_name": lora.weight_file_name,
                "weight": float(lora.weight),
                "target": lora.target,
            }
            for lora in args.loras
        ]

    encoded = json.dumps(
        model_args, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
