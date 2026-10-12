from types import SimpleNamespace

import torch

from sd_task.cache import MemoryModelCache
from sd_task.config import Config
from sd_task.task_args.image_task import ImageTaskArgs
from sd_task.task_runner.image_task import ImageTaskResult
from sd_task.task_runner.image_task import image_task as image_module
from sd_task.task_runner.image_task.result import resolve_execution_dtype


class FakePipeline:
    def __init__(self, dtype: torch.dtype):
        self.transformer = torch.nn.Linear(1, 1, bias=False).to(dtype=dtype)
        self.calls = 0

    def __call__(
        self,
        prompt,
        width,
        height,
        num_inference_steps,
        generator,
        num_images_per_prompt,
        guidance_scale=None,
    ):
        self.calls += 1
        return SimpleNamespace(images=[object() for _ in range(num_images_per_prompt)])


def make_args(*, dtype: str = "auto", num_images_per_prompt: int = 1) -> ImageTaskArgs:
    return ImageTaskArgs.model_validate(
        {
            "version": "3.7.0",
            "base_model": {"name": "test/model"},
            "dtype": dtype,
            "prompt": "test",
            "task_config": {"num_images_per_prompt": num_images_per_prompt},
        }
    )


def configure_fake_image_task(monkeypatch, loaded_pipelines: list[FakePipeline]):
    monkeypatch.setattr(image_module.utils, "get_gpu_info", lambda: None)
    monkeypatch.setattr(image_module.utils, "get_accelerator", lambda: "cpu")
    monkeypatch.setattr(
        image_module, "resolve_models_from_cache", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        image_module,
        "resolve_pipeline_class",
        lambda *args, **kwargs: FakePipeline,
    )

    def load_pipeline(*args, **kwargs):
        torch_dtype = kwargs.get("torch_dtype")
        if torch_dtype is None and len(args) >= 4:
            torch_dtype = args[3]
        pipeline = FakePipeline(torch_dtype or torch.float32)
        loaded_pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(image_module, "load_pipeline", load_pipeline)
    monkeypatch.setattr(
        image_module, "move_pipeline_to_accelerator", lambda pipeline: pipeline
    )


def test_fp16_reports_loaded_transformer_dtype_and_remains_list_compatible(
    monkeypatch,
):
    loaded_pipelines: list[FakePipeline] = []
    configure_fake_image_task(monkeypatch, loaded_pipelines)

    result = image_module.run_image_task(
        make_args(dtype="float16", num_images_per_prompt=2),
        config=Config(deterministic=False, local_files_only=True),
    )

    assert isinstance(result, ImageTaskResult)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result.execution_dtype == "float16"
    assert len(loaded_pipelines) == 1
    assert loaded_pipelines[0].calls == 1


def test_cache_hit_reports_dtype_from_cached_transformer(monkeypatch):
    loaded_pipelines: list[FakePipeline] = []
    configure_fake_image_task(monkeypatch, loaded_pipelines)
    cache = MemoryModelCache()
    args = make_args(dtype="float16")
    config = Config(deterministic=False, local_files_only=True)

    first_result = image_module.run_image_task(
        args,
        config=config,
        model_cache=cache,
    )
    loaded_pipelines[0].transformer.to(dtype=torch.bfloat16)
    cached_result = image_module.run_image_task(
        args,
        config=config,
        model_cache=cache,
    )

    assert first_result.execution_dtype == "float16"
    assert cached_result.execution_dtype == "bfloat16"
    assert len(loaded_pipelines) == 1


def test_execution_dtype_requires_transformer():
    pipeline = SimpleNamespace(
        transformer=torch.nn.Linear(1, 1, bias=False).to(dtype=torch.float32)
    )

    assert resolve_execution_dtype(pipeline) == "float32"
