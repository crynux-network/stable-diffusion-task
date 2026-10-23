import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sd_task.task_args.image_task import ImageTaskArgs, QuantizationConfig
from sd_task.task_runner.image_task import dtype as dtype_module
from sd_task.task_runner.image_task import pipeline as pipeline_module
from sd_task.task_runner.image_task.dtype import resolve_image_dtype
from sd_task.task_runner.image_task.errors import TaskInvalid
from sd_task.task_runner.image_task.key import generate_model_key
from sd_task.task_runner.image_task.quantization import bitsandbytes as bnb_module
from sd_task.task_runner.image_task.quantization import gguf as gguf_module


def make_args(**updates) -> ImageTaskArgs:
    data = {
        "version": "3.7.0",
        "base_model": {"name": "org/base"},
        "prompt": "test",
        "task_config": {},
    }
    data.update(updates)
    return ImageTaskArgs.model_validate(data)


def test_resolve_image_dtype_concrete_values():
    for name, expected in (
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32),
    ):
        resolved_name, resolved = resolve_image_dtype(
            name,
            base_model_name="org/base",
            base_model_revision=None,
            cache_dir="/cache",
            local_files_only=True,
        )
        assert resolved_name == name
        assert resolved is expected


def test_resolve_image_dtype_auto_reads_transformer_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"torch_dtype": "bfloat16"}), encoding="utf-8")
    monkeypatch.setattr(
        dtype_module, "hf_hub_download", lambda **kwargs: str(config_path)
    )

    name, resolved = resolve_image_dtype(
        "auto",
        base_model_name="org/base",
        base_model_revision=None,
        cache_dir="/cache",
        local_files_only=True,
    )
    assert name == "bfloat16"
    assert resolved is torch.bfloat16


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"torch_dtype": None},
        {"torch_dtype": "auto"},
    ],
)
def test_resolve_image_dtype_auto_falls_back_to_float32(
    monkeypatch, tmp_path, payload
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        dtype_module, "hf_hub_download", lambda **kwargs: str(config_path)
    )

    name, resolved = resolve_image_dtype(
        "auto",
        base_model_name="org/base",
        base_model_revision=None,
        cache_dir="/cache",
        local_files_only=True,
    )
    assert name == "float32"
    assert resolved is torch.float32


def test_resolve_image_dtype_auto_rejects_illegal_config_value(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"torch_dtype": "float64"}), encoding="utf-8")
    monkeypatch.setattr(
        dtype_module, "hf_hub_download", lambda **kwargs: str(config_path)
    )

    with pytest.raises(TaskInvalid, match="Unsupported transformer config"):
        resolve_image_dtype(
            "auto",
            base_model_name="org/base",
            base_model_revision=None,
            cache_dir="/cache",
            local_files_only=True,
        )


def test_load_pipeline_and_gguf_share_resolved_dtype(monkeypatch):
    captured = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            captured["pipeline_dtype"] = kwargs.get("dtype")
            return SimpleNamespace(name=name)

    class FakeTransformer:
        @classmethod
        def from_single_file(cls, path, **kwargs):
            captured["gguf_dtype"] = kwargs.get("dtype")
            captured["gguf_compute"] = kwargs["quantization_config"].compute_dtype
            return MagicMock()

    monkeypatch.setattr(
        "sd_task.task_runner.image_task.quantization.gguf.pipeline_config_from_base",
        lambda *args, **kwargs: {"transformer": ["diffusers", "FakeTransformer"]},
    )
    monkeypatch.setattr(
        "sd_task.task_runner.image_task.quantization.gguf."
        "component_class_from_pipeline_config",
        lambda config, name: FakeTransformer,
    )
    monkeypatch.setattr(
        gguf_module,
        "_resolve_component_config_dir",
        lambda **kwargs: "/cache/transformer",
    )
    monkeypatch.setattr(
        gguf_module,
        "_materialize_unquantized_gguf_parameters",
        lambda model, compute_dtype: model,
    )

    args = make_args(
        dtype="bfloat16",
        components={
            "transformer": {
                "format": "gguf",
                "repo": "org/gguf",
                "format_kwargs": {"filename": "model.gguf"},
            }
        },
    )
    pipeline_module.load_pipeline(
        FakePipeline,
        args,
        cache_dir="/cache",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        component_paths={"transformer": "/cache/model.gguf"},
    )

    assert captured["pipeline_dtype"] is torch.bfloat16
    assert captured["gguf_dtype"] is torch.bfloat16
    assert captured["gguf_compute"] is torch.bfloat16


def test_bnb_default_compute_dtype_follows_resolved_dtype(monkeypatch):
    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(
        bnb_module,
        "_accepted_params",
        lambda config_cls: {
            "load_in_4bit",
            "load_in_8bit",
            "bnb_4bit_compute_dtype",
            "llm_int8_enable_fp32_cpu_offload",
            "llm_int8_has_fp16_weight",
        },
    )

    import sys

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(BitsAndBytesConfig=FakeBitsAndBytesConfig),
    )

    quant = QuantizationConfig.model_validate(
        {
            "quant_method": "bitsandbytes",
            "components_to_quantize": ["transformer"],
            "quant_kwargs": {"load_in_4bit": True},
        }
    )
    kwargs = bnb_module._normalize_quant_kwargs(
        quant, make_args(dtype="bfloat16")
    )
    assert kwargs["bnb_4bit_compute_dtype"] is torch.bfloat16


def test_resolved_dtype_participates_in_cache_key():
    class Pipeline:
        pass

    auto_key = generate_model_key(make_args(dtype="auto"), Pipeline)
    bf16_key = generate_model_key(make_args(dtype="bfloat16"), Pipeline)
    fp16_key = generate_model_key(make_args(dtype="float16"), Pipeline)

    assert auto_key != bf16_key
    assert bf16_key != fp16_key


def test_load_pipeline_always_passes_concrete_dtype():
    captured = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(name=name)

    pipeline_module.load_pipeline(
        FakePipeline,
        make_args(dtype="float16"),
        cache_dir="/cache",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    assert captured["dtype"] is torch.float16
