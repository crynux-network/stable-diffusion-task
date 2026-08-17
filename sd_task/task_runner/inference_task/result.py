from collections.abc import Iterable
from typing import Any

from PIL import Image


class InferenceTaskResult(list[Image.Image]):
    def __init__(self, images: Iterable[Image.Image], execution_dtype: str):
        super().__init__(images)
        self.execution_dtype = execution_dtype


def resolve_execution_dtype(pipeline: Any) -> str:
    denoiser = getattr(pipeline, "unet", None)
    if denoiser is None:
        denoiser = getattr(pipeline, "transformer", None)
    if denoiser is None:
        raise RuntimeError("The pipeline does not expose a primary denoising model")

    try:
        parameter = next(denoiser.parameters())
    except StopIteration as e:
        raise RuntimeError("The primary denoising model has no parameters") from e

    return str(parameter.dtype).removeprefix("torch.")
