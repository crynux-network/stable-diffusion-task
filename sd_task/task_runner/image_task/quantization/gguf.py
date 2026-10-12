from __future__ import annotations

import os
from typing import Any, Type

import torch
from diffusers import DiffusionPipeline, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download

from sd_task.task_args.image_task import ComponentSource
from sd_task.task_runner.image_task.errors import TaskInvalid

from .errors import UnsupportedComponentFormat

SUPPORTED_GGUF_TENSOR_TYPES = frozenset(
    {
        "BF16",
        "Q4_0",
        "Q4_1",
        "Q5_0",
        "Q5_1",
        "Q8_0",
        "Q2_K",
        "Q3_K",
        "Q4_K",
        "Q5_K",
        "Q6_K",
    }
)


def resolve_gguf_filename(source: ComponentSource) -> str:
    if source.format != "gguf":
        raise UnsupportedComponentFormat(source.format)
    filename = source.format_kwargs.get("filename")
    if not isinstance(filename, str) or not filename:
        raise TaskInvalid("GGUF component requires format_kwargs.filename")
    return filename


def load_gguf_transformer(
    *,
    pipeline_config: dict[str, Any],
    source: ComponentSource,
    base_model_name: str,
    base_model_revision: str | None,
    cache_dir: str,
    torch_dtype: torch.dtype | None,
    local_files_only: bool,
    component_path: str,
) -> Any:
    if source.format != "gguf":
        raise UnsupportedComponentFormat(source.format)

    transformer_entry = pipeline_config.get("transformer")
    if not isinstance(transformer_entry, (list, tuple)) or len(transformer_entry) != 2:
        raise TaskInvalid("Pipeline model_index.json does not declare transformer")

    library_name, class_name = transformer_entry
    if not isinstance(library_name, str) or not isinstance(class_name, str):
        raise TaskInvalid("Pipeline transformer class declaration is invalid")

    transformer_cls = _resolve_component_class(library_name, class_name)
    if not hasattr(transformer_cls, "from_single_file"):
        raise TaskInvalid(
            f"Transformer class {class_name} does not support from_single_file"
        )

    config_dir = _resolve_component_config_dir(
        base_model_name=base_model_name,
        base_model_revision=base_model_revision,
        component_name="transformer",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    compute_dtype = torch_dtype if torch_dtype is not None else torch.float32
    load_args: dict[str, Any] = {
        "config": config_dir,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "quantization_config": GGUFQuantizationConfig(compute_dtype=compute_dtype),
        "dtype": compute_dtype,
    }
    transformer = transformer_cls.from_single_file(component_path, **load_args)
    return _materialize_unquantized_gguf_parameters(transformer, compute_dtype)


def _materialize_unquantized_gguf_parameters(
    model: Any, compute_dtype: torch.dtype
) -> Any:
    """Convert F32/F16/BF16 GGUFParameter tensors into plain Parameters.

    Diffusers keeps unquantized GGUF tensors as byte-backed GGUFParameter
    values. Modules such as RMSNorm call ``weight.float()`` directly, which
    keeps the byte storage shape (for BF16, logical N becomes 2N) and breaks
    inference. Linear layers avoid this by dequantizing through GGUFLinear.
    """
    from diffusers.quantizers.gguf.utils import (
        GGUFParameter,
        UNQUANTIZED_TYPES,
        dequantize_gguf_tensor,
    )

    for module in model.modules():
        for param_name, param in list(module._parameters.items()):
            if param is None or not isinstance(param, GGUFParameter):
                continue
            if param.quant_type not in UNQUANTIZED_TYPES:
                continue
            dequantized = dequantize_gguf_tensor(param).to(dtype=compute_dtype)
            module._parameters[param_name] = torch.nn.Parameter(
                dequantized, requires_grad=False
            )
    return model


def _resolve_component_config_dir(
    *,
    base_model_name: str,
    base_model_revision: str | None,
    component_name: str,
    cache_dir: str,
    local_files_only: bool,
) -> str:
    download_args: dict[str, Any] = {
        "repo_id": base_model_name,
        "filename": f"{component_name}/config.json",
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if base_model_revision is not None:
        download_args["revision"] = base_model_revision
    config_file = hf_hub_download(**download_args)
    return os.path.dirname(config_file)


def _resolve_component_class(library_name: str, class_name: str) -> Type[Any]:
    if library_name == "diffusers":
        module = __import__("diffusers")
    elif library_name == "transformers":
        module = __import__("transformers")
    else:
        raise TaskInvalid(f"Unsupported component library: {library_name}")
    component_cls = getattr(module, class_name, None)
    if component_cls is None:
        raise TaskInvalid(f"Component class {class_name} is unavailable")
    return component_cls


def pipeline_config_from_base(
    model_name: str,
    cache_dir: str,
    local_files_only: bool,
    revision: str | None = None,
) -> dict[str, Any]:
    load_args: dict[str, Any] = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        load_args["revision"] = revision
    config = DiffusionPipeline.load_config(model_name, **load_args)
    if not isinstance(config, dict):
        raise TaskInvalid("Pipeline config is invalid")
    return config
