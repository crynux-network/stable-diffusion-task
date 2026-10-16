from collections.abc import Iterable
from typing import Any

import torch
from PIL import Image


class ImageTaskResult(list[Image.Image]):
    def __init__(
        self,
        images: Iterable[Image.Image],
        execution_dtype: str,
        execution_quantization: dict[str, Any] | None = None,
    ):
        super().__init__(images)
        self.execution_dtype = execution_dtype
        self.execution_quantization = execution_quantization


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _floating_parameter_dtype(module: Any) -> str | None:
    dtype = getattr(module, "dtype", None)
    if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
        return _dtype_name(dtype)
    try:
        for parameter in module.parameters():
            if parameter.is_floating_point():
                return _dtype_name(parameter.dtype)
    except StopIteration:
        return None
    return None


def _compute_dtype_from_quantization(quantization: dict[str, Any] | None) -> str | None:
    if quantization is None:
        return None
    effective = quantization.get("effective_config")
    if not isinstance(effective, dict):
        return None
    if effective.get("quant_method") != "bitsandbytes":
        return None
    kwargs = effective.get("quant_kwargs")
    if not isinstance(kwargs, dict):
        return None
    compute = kwargs.get("bnb_4bit_compute_dtype")
    if compute is None:
        return None
    if isinstance(compute, torch.dtype):
        return _dtype_name(compute)
    name = str(compute).removeprefix("torch.")
    if name in {"float16", "bfloat16", "float32", "float", "half"}:
        if name == "float":
            return "float32"
        if name == "half":
            return "float16"
        return name
    return None


def resolve_execution_dtype(
    pipeline: Any,
    *,
    quantization: dict[str, Any] | None = None,
) -> str:
    """Return the floating compute dtype used by the loaded pipeline.

    Quantized packed weight dtypes such as uint8 MUST NOT be reported.
    """
    from_quant = _compute_dtype_from_quantization(quantization)
    if from_quant is not None:
        return from_quant

    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError("The pipeline does not expose a transformer")

    floating = _floating_parameter_dtype(transformer)
    if floating is not None:
        return floating

    raise RuntimeError(
        "The pipeline transformer has no floating-point parameters to report "
        "as execution dtype"
    )
