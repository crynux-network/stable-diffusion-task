from types import SimpleNamespace

import torch

from sd_task.cache import MemoryModelCache
from sd_task.config import Config
from sd_task.task_args.inference_task import InferenceTaskArgs
from sd_task.task_runner.inference_task import InferenceTaskResult
from sd_task.task_runner.inference_task import inference_task as inference_module
from sd_task.task_runner.inference_task.result import resolve_execution_dtype


class FakePipeline:
    def __init__(self, dtype: torch.dtype):
        self.unet = torch.nn.Linear(1, 1, bias=False).to(dtype=dtype)

    def to(self, device: str):
        return self

    def __call__(self, **kwargs):
        return SimpleNamespace(images=[object()])


def make_args(*, variant: str | None = None) -> InferenceTaskArgs:
    return InferenceTaskArgs.model_validate(
        {
            "version": "3.0.0",
            "base_model": {"name": "test/model", "variant": variant},
            "prompt": "test",
            "task_config": {"num_images": 1},
        }
    )


def configure_fake_inference(monkeypatch, loaded_pipelines: list[FakePipeline]):
    monkeypatch.setattr(inference_module.utils, "get_gpu_info", lambda: None)
    monkeypatch.setattr(inference_module.utils, "get_accelerator", lambda: "cpu")
    monkeypatch.setattr(
        inference_module, "resolve_models_from_cache", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        inference_module, "add_scheduler_pipeline_args", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        inference_module,
        "get_pipeline_call_args",
        lambda *args, **kwargs: {"generator": object()},
    )

    def from_pretrained(model_id: str, **kwargs):
        pipeline = FakePipeline(kwargs["torch_dtype"] or torch.float32)
        loaded_pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(
        inference_module.AutoPipelineForText2Image,
        "from_pretrained",
        from_pretrained,
    )


def test_fp16_variant_reports_loaded_denoiser_dtype_and_remains_list_compatible(
    monkeypatch,
):
    loaded_pipelines: list[FakePipeline] = []
    configure_fake_inference(monkeypatch, loaded_pipelines)

    result = inference_module.run_inference_task(
        make_args(variant="fp16"),
        config=Config(deterministic=False, local_files_only=True),
    )

    assert isinstance(result, InferenceTaskResult)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result.execution_dtype == "float16"
    assert len(loaded_pipelines) == 1


def test_cache_hit_reports_dtype_from_cached_denoiser(monkeypatch):
    loaded_pipelines: list[FakePipeline] = []
    configure_fake_inference(monkeypatch, loaded_pipelines)
    cache = MemoryModelCache()
    args = make_args(variant="fp16")
    config = Config(deterministic=False, local_files_only=True)

    first_result = inference_module.run_inference_task(
        args,
        config=config,
        model_cache=cache,
    )
    loaded_pipelines[0].unet.to(dtype=torch.bfloat16)
    cached_result = inference_module.run_inference_task(
        args,
        config=config,
        model_cache=cache,
    )

    assert first_result.execution_dtype == "float16"
    assert cached_result.execution_dtype == "bfloat16"
    assert len(loaded_pipelines) == 1


def test_transformer_is_used_when_pipeline_has_no_unet():
    pipeline = SimpleNamespace(
        transformer=torch.nn.Linear(1, 1, bias=False).to(dtype=torch.float32)
    )

    assert resolve_execution_dtype(pipeline) == "float32"
