from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sd_task.task_runner.image_task.result import (
    _compute_dtype_from_quantization,
    _floating_parameter_dtype,
)


@dataclass(frozen=True)
class VideoTaskResult:
    output_path: str
    score: bytes
    num_frames: int
    width: int
    height: int
    fps: float
    has_audio: bool
    execution_dtype: str
    execution_quantization: dict[str, Any] | None = None

    @property
    def score_hex(self) -> str:
        return "0x" + self.score.hex()


def resolve_video_execution_dtype(
    pipeline: Any,
    *,
    quantization: dict[str, Any] | None = None,
) -> str:
    from_quant = _compute_dtype_from_quantization(quantization)
    if from_quant is not None:
        return from_quant

    for attr in ("transformer", "transformer_2", "transformer_ref"):
        denoiser = getattr(pipeline, attr, None)
        if denoiser is None:
            continue
        floating = _floating_parameter_dtype(denoiser)
        if floating is not None:
            return floating

    raise RuntimeError(
        "The video pipeline exposes no floating-point denoiser parameters to "
        "report as execution dtype"
    )
