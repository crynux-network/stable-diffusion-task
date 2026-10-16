import sys
from types import ModuleType

import pytest

from sd_task.task_args.image_task import ImageTaskArgs
from sd_task.task_runner.image_task.quantization import bitsandbytes as bnb_mod
from sd_task.task_runner.image_task.quantization.errors import (
    ImageQuantizationRuntimeUnsupported,
)


def make_args() -> ImageTaskArgs:
    return ImageTaskArgs.model_validate(
        {
            "version": "3.7.0",
            "base_model": {"name": "test/model"},
            "dtype": "bfloat16",
            "prompt": "test",
            "task_config": {"num_images_per_prompt": 1},
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "components_to_quantize": ["transformer"],
                "quant_kwargs": {"load_in_4bit": True},
            },
        }
    )


def test_bitsandbytes_resolve_does_not_platform_block_on_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    class FakePipelineQuantizationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    quantizers = ModuleType("diffusers.quantizers")
    quantizers.PipelineQuantizationConfig = FakePipelineQuantizationConfig
    monkeypatch.setitem(sys.modules, "bitsandbytes", ModuleType("bitsandbytes"))
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", quantizers)

    config = bnb_mod.resolve_pipeline_quantization_config(
        make_args().quantization_config, make_args()
    )
    assert isinstance(config, FakePipelineQuantizationConfig)
    assert config.kwargs["quant_backend"] == "bitsandbytes_4bit"


def test_bitsandbytes_unavailable_still_raises(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "bitsandbytes" or name.startswith("bitsandbytes"):
            raise ImportError("no bitsandbytes")
        if name == "diffusers.quantizers" or name.startswith("diffusers.quantizers"):
            raise ImportError("no diffusers quantizers")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImageQuantizationRuntimeUnsupported):
        bnb_mod.resolve_pipeline_quantization_config(
            make_args().quantization_config, make_args()
        )
