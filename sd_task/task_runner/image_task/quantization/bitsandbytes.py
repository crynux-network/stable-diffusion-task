from __future__ import annotations

import inspect
from typing import Any

import torch

from sd_task.task_args.image_task import ImageTaskArgs, QuantizationConfig

from .errors import ImageQuantizationRuntimeUnsupported


def _accepted_params(config_cls: type) -> set[str]:
    signature = inspect.signature(config_cls.__init__)
    return {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "half": torch.float16,
    }
    key = name.removeprefix("torch.")
    if key not in mapping:
        raise ValueError(f"Unsupported dtype value: {name}")
    return mapping[key]


def _normalize_quant_kwargs(
    quantization_config: QuantizationConfig,
    args: ImageTaskArgs,
) -> dict[str, Any]:
    kwargs = dict(quantization_config.quant_kwargs)
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImageQuantizationRuntimeUnsupported(
            "BitsAndBytesConfig is unavailable"
        ) from exc

    accepted = _accepted_params(BitsAndBytesConfig)
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise ValueError(
            f"BitsAndBytesConfig does not accept fields: {', '.join(unknown)}"
        )

    if kwargs.get("load_in_4bit") and "bnb_4bit_compute_dtype" not in kwargs:
        kwargs["bnb_4bit_compute_dtype"] = _torch_dtype(args.dtype)
    elif "bnb_4bit_compute_dtype" in kwargs and isinstance(
        kwargs["bnb_4bit_compute_dtype"], str
    ):
        kwargs["bnb_4bit_compute_dtype"] = _torch_dtype(
            kwargs["bnb_4bit_compute_dtype"]
        )

    kwargs["llm_int8_enable_fp32_cpu_offload"] = False
    kwargs["llm_int8_has_fp16_weight"] = False
    return kwargs


def resolve_pipeline_quantization_config(
    quantization_config: QuantizationConfig,
    args: ImageTaskArgs,
) -> Any:
    try:
        import bitsandbytes  # noqa: F401
        from diffusers.quantizers import PipelineQuantizationConfig
    except ImportError as exc:
        raise ImageQuantizationRuntimeUnsupported(
            "Diffusers PipelineQuantizationConfig or bitsandbytes is unavailable"
        ) from exc

    kwargs = _normalize_quant_kwargs(quantization_config, args)
    load_in_4bit = bool(kwargs.get("load_in_4bit"))
    load_in_8bit = bool(kwargs.get("load_in_8bit"))
    if load_in_4bit == load_in_8bit:
        raise ValueError(
            "bitsandbytes quant_kwargs must set exactly one of "
            "load_in_4bit or load_in_8bit"
        )
    quant_backend = "bitsandbytes_4bit" if load_in_4bit else "bitsandbytes_8bit"
    return PipelineQuantizationConfig(
        quant_backend=quant_backend,
        quant_kwargs=kwargs,
        components_to_quantize=list(quantization_config.components_to_quantize),
    )


def effective_bitsandbytes_identity(
    quantization_config: QuantizationConfig,
    args: ImageTaskArgs,
) -> dict[str, Any]:
    kwargs = _normalize_quant_kwargs(quantization_config, args)
    normalized: dict[str, Any] = {}
    for key, value in sorted(kwargs.items(), key=lambda item: str(item[0])):
        if isinstance(value, torch.dtype):
            normalized[key] = str(value).removeprefix("torch.")
        else:
            normalized[key] = value
    return {
        "quant_method": "bitsandbytes",
        "components_to_quantize": list(
            quantization_config.components_to_quantize
        ),
        "quant_kwargs": normalized,
    }
