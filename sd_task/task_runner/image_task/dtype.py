from __future__ import annotations

import json
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from sd_task.task_runner.image_task.errors import TaskInvalid

_CONCRETE_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

__all__ = [
    "resolve_image_dtype",
    "torch_dtype_from_name",
]


def torch_dtype_from_name(name: str) -> torch.dtype:
    key = name.removeprefix("torch.").lower().strip()
    if key not in _CONCRETE_DTYPES:
        raise TaskInvalid(f"Unsupported dtype value: {name}")
    return _CONCRETE_DTYPES[key]


def resolve_image_dtype(
    dtype: str,
    *,
    base_model_name: str,
    base_model_revision: str | None,
    cache_dir: str,
    local_files_only: bool,
) -> tuple[str, torch.dtype]:
    """Resolve TaskArgs dtype to a concrete name and torch.dtype before loading."""
    normalized = dtype.lower().strip()
    if normalized in _CONCRETE_DTYPES:
        return normalized, _CONCRETE_DTYPES[normalized]
    if normalized != "auto":
        raise TaskInvalid(f"Unsupported dtype value: {dtype}")

    config_dtype = _read_transformer_config_dtype(
        base_model_name=base_model_name,
        base_model_revision=base_model_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if config_dtype is None:
        return "float32", torch.float32
    return config_dtype, _CONCRETE_DTYPES[config_dtype]


def _read_transformer_config_dtype(
    *,
    base_model_name: str,
    base_model_revision: str | None,
    cache_dir: str,
    local_files_only: bool,
) -> str | None:
    download_args: dict[str, Any] = {
        "repo_id": base_model_name,
        "filename": "transformer/config.json",
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if base_model_revision is not None:
        download_args["revision"] = base_model_revision
    try:
        config_path = hf_hub_download(**download_args)
    except Exception as exc:
        raise TaskInvalid(
            "Failed to read base pipeline transformer config for dtype=auto"
        ) from exc

    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        raise TaskInvalid(
            "Base pipeline transformer config is not valid JSON"
        ) from exc

    if not isinstance(config, dict):
        raise TaskInvalid("Base pipeline transformer config is invalid")

    raw = config.get("torch_dtype")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TaskInvalid(
            f"Unsupported transformer config torch_dtype value: {raw!r}"
        )
    normalized = raw.removeprefix("torch.").lower().strip()
    if normalized == "" or normalized == "auto":
        return None
    if normalized not in _CONCRETE_DTYPES:
        raise TaskInvalid(
            f"Unsupported transformer config torch_dtype value: {raw}"
        )
    return normalized
