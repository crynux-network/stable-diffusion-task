from __future__ import annotations

import json
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from sd_task.task_args.video_task import VideoInputMode
from sd_task.task_runner.image_task.dtype import _CONCRETE_DTYPES
from sd_task.task_runner.image_task.errors import TaskInvalid

__all__ = ["resolve_video_dtype"]


def _candidate_denoiser_subfolders(input_mode: VideoInputMode) -> tuple[str, ...]:
    if input_mode == "r2v":
        return ("transformer_ref", "transformer", "transformer_2")
    return ("transformer", "transformer_2", "transformer_ref")


def resolve_video_dtype(
    dtype: str,
    *,
    base_model_name: str,
    base_model_revision: str | None,
    input_mode: VideoInputMode,
    cache_dir: str,
    local_files_only: bool,
) -> tuple[str, torch.dtype]:
    """Resolve VideoTaskArgs dtype to a concrete name and torch.dtype before loading."""
    normalized = dtype.lower().strip()
    if normalized in _CONCRETE_DTYPES:
        return normalized, _CONCRETE_DTYPES[normalized]
    if normalized != "auto":
        raise TaskInvalid(f"Unsupported dtype value: {dtype}")

    config_dtype = _read_denoiser_config_dtype(
        base_model_name=base_model_name,
        base_model_revision=base_model_revision,
        input_mode=input_mode,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if config_dtype is None:
        return "float32", torch.float32
    return config_dtype, _CONCRETE_DTYPES[config_dtype]


def _read_denoiser_config_dtype(
    *,
    base_model_name: str,
    base_model_revision: str | None,
    input_mode: VideoInputMode,
    cache_dir: str,
    local_files_only: bool,
) -> str | None:
    config_path: str | None = None
    last_exc: Exception | None = None
    for subfolder in _candidate_denoiser_subfolders(input_mode):
        download_args: dict[str, Any] = {
            "repo_id": base_model_name,
            "filename": f"{subfolder}/config.json",
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if base_model_revision is not None:
            download_args["revision"] = base_model_revision
        try:
            config_path = hf_hub_download(**download_args)
            break
        except Exception as exc:
            last_exc = exc

    if config_path is None:
        raise TaskInvalid(
            "Failed to read base pipeline denoiser config for dtype=auto"
        ) from last_exc

    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        raise TaskInvalid(
            "Base pipeline denoiser config is not valid JSON"
        ) from exc

    if not isinstance(config, dict):
        raise TaskInvalid("Base pipeline denoiser config is invalid")

    raw = config.get("torch_dtype")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TaskInvalid(
            f"Unsupported denoiser config torch_dtype value: {raw!r}"
        )
    normalized = raw.removeprefix("torch.").lower().strip()
    if normalized == "" or normalized == "auto":
        return None
    if normalized not in _CONCRETE_DTYPES:
        raise TaskInvalid(
            f"Unsupported denoiser config torch_dtype value: {raw}"
        )
    return normalized
