from collections.abc import Iterable
from typing import Any

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


def resolve_execution_dtype(pipeline: Any) -> str:
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError("The pipeline does not expose a transformer")

    try:
        parameter = next(transformer.parameters())
    except StopIteration as e:
        raise RuntimeError("The pipeline transformer has no parameters") from e

    return str(parameter.dtype).removeprefix("torch.")
