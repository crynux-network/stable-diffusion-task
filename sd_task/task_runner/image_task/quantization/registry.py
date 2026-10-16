from __future__ import annotations

import sys
from typing import Any

from sd_task.task_args.image_task import (
    ComponentSource,
    ImageTaskArgs,
)

from .bitsandbytes import (
    effective_bitsandbytes_identity,
    resolve_pipeline_quantization_config,
)
from .diffusers_component import (
    DIFFUSERS_COMPONENT_FORMAT,
    resolve_diffusers_subfolder,
)
from .errors import (
    ImageQuantizationRuntimeUnsupported,
    UnsupportedComponentFormat,
    UnsupportedImageQuantizationBackend,
)
from .gguf import resolve_gguf_filename


def resolve_image_quantization(
    args: ImageTaskArgs,
) -> dict[str, Any] | None:
    config = args.quantization_config
    if config is None:
        return None
    if config.quant_method == "bitsandbytes":
        pipeline_quant_config = resolve_pipeline_quantization_config(config, args)
        return {
            "quant_method": "bitsandbytes",
            "pipeline_quantization_config": pipeline_quant_config,
            "components_to_quantize": list(config.components_to_quantize),
            "effective_config": effective_bitsandbytes_identity(config, args),
        }
    raise UnsupportedImageQuantizationBackend(config.quant_method)


def resolve_component_loader(source: ComponentSource) -> str:
    if source.format == "gguf":
        resolve_gguf_filename(source)
        return "gguf"
    if source.format == DIFFUSERS_COMPONENT_FORMAT:
        resolve_diffusers_subfolder(source)
        return DIFFUSERS_COMPONENT_FORMAT
    raise UnsupportedComponentFormat(source.format)


def assert_nunchaku_platform_supported() -> None:
    if sys.platform == "win32":
        raise ImageQuantizationRuntimeUnsupported(
            "Nunchaku Lite kernels are unavailable on Windows"
        )
