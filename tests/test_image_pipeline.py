import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from sd_task.task_args import ImageTaskArgs
from sd_task.task_runner.image_task import pipeline as pipeline_module
from sd_task.task_runner.image_task.errors import (
    TaskInvalid,
    wrap_model_load_error,
)
from sd_task.task_runner.image_task.guidance import add_guidance_call_arg
from sd_task.task_runner.image_task.key import generate_model_key


class NativeTextPipeline:
    def __call__(self, prompt):
        raise NotImplementedError


class NativeUnifiedPipeline:
    def __call__(self, prompt, image: Image.Image | list[Image.Image] | None = None):
        raise NotImplementedError


class ImageToImagePipeline:
    def __call__(self, prompt, image: Image.Image):
        raise NotImplementedError


def make_args(**updates) -> ImageTaskArgs:
    data = {
        "version": "3.7.0",
        "base_model": {"name": "test/model"},
        "prompt": "test",
        "task_config": {},
    }
    data.update(updates)
    return ImageTaskArgs.model_validate(data)


def image_data_url() -> str:
    buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def configure_native_class(monkeypatch, class_name, pipeline_class):
    import diffusers

    monkeypatch.setattr(diffusers, class_name, pipeline_class, raising=False)
    monkeypatch.setattr(
        pipeline_module.DiffusionPipeline,
        "load_config",
        classmethod(lambda cls, *args, **kwargs: {"_class_name": class_name}),
    )


def test_text_to_image_uses_checkpoint_native_class(monkeypatch):
    configure_native_class(monkeypatch, "TestPipeline", NativeTextPipeline)

    resolved = pipeline_module.resolve_pipeline_class(
        "test/model", "cache", True, False
    )

    assert resolved is NativeTextPipeline


def test_image_input_uses_auto_pipeline_mapping(monkeypatch):
    configure_native_class(monkeypatch, "TestPipeline", NativeTextPipeline)
    monkeypatch.setattr(
        pipeline_module,
        "_get_task_class",
        lambda *args, **kwargs: ImageToImagePipeline,
    )

    resolved = pipeline_module.resolve_pipeline_class("test/model", "cache", True, True)

    assert resolved is ImageToImagePipeline


def test_image_input_prefers_native_unified_pipeline(monkeypatch):
    configure_native_class(monkeypatch, "TestPipeline", NativeUnifiedPipeline)

    def fail_if_auto_mapping_is_used(*args, **kwargs):
        raise AssertionError("AutoPipeline mapping must not run")

    monkeypatch.setattr(
        pipeline_module, "_get_task_class", fail_if_auto_mapping_is_used
    )

    resolved = pipeline_module.resolve_pipeline_class("test/model", "cache", True, True)

    assert resolved is NativeUnifiedPipeline


def test_unsupported_image_input_fails_before_loading_weights(monkeypatch):
    configure_native_class(monkeypatch, "TestPipeline", NativeTextPipeline)
    monkeypatch.setattr(
        pipeline_module, "_get_task_class", lambda *args, **kwargs: None
    )
    load_calls = 0

    def from_pretrained(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1

    monkeypatch.setattr(
        NativeTextPipeline, "from_pretrained", from_pretrained, raising=False
    )

    with pytest.raises(TaskInvalid):
        pipeline_module.resolve_pipeline_class("test/model", "cache", True, True)

    assert load_calls == 0


def test_load_pipeline_calls_only_resolved_class_once():
    calls = []

    class LoadingPipeline:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            return object()

    result = pipeline_module.load_pipeline(
        LoadingPipeline,
        make_args(),
        cache_dir="cache",
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    assert result is not None
    assert len(calls) == 1
    assert "scheduler" not in calls[0][1]
    assert calls[0][1]["dtype"] is torch.float16
    assert calls[0][0] == "test/model"


def test_lora_uses_standard_pipeline_interfaces():
    class LoraPipeline:
        def __init__(self):
            self.loaded = None
            self.adapters = None

        def load_lora_weights(self, model, **kwargs):
            self.loaded = (model, kwargs)

        def set_adapters(self, names, adapter_weights):
            self.adapters = (names, adapter_weights)

    pipeline = LoraPipeline()

    pipeline_module.load_lora(
        pipeline,
        model="test/lora",
        weight_file_name="weights.safetensors",
        weight=0.8,
        cache_dir="cache",
        local_files_only=True,
    )

    assert pipeline.loaded == (
        "test/lora",
        {
            "adapter_name": "task",
            "cache_dir": "cache",
            "local_files_only": True,
            "weight_name": "weights.safetensors",
        },
    )
    assert pipeline.adapters == (["task"], [0.8])


def test_incompatible_lora_is_rejected_as_invalid_task():
    class IncompatibleLoraPipeline:
        def load_lora_weights(self, model, **kwargs):
            raise ValueError("incompatible")

        def set_adapters(self, names, adapter_weights):
            raise AssertionError("set_adapters must not be called")

    with pytest.raises(TaskInvalid):
        pipeline_module.load_lora(
            IncompatibleLoraPipeline(),
            model="test/lora",
            weight_file_name="",
            weight=1,
            cache_dir="cache",
            local_files_only=True,
        )


def test_invalid_model_capability_is_not_reclassified():
    with pytest.raises(TaskInvalid):
        with wrap_model_load_error(local_files_only=False):
            raise TaskInvalid("unsupported")


class BothGuidancePipeline:
    def __init__(self, guidance_embeds):
        self.transformer = SimpleNamespace(
            config=SimpleNamespace(guidance_embeds=guidance_embeds)
        )

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        guidance_scale=None,
        true_cfg_scale=1,
    ):
        raise NotImplementedError


class TrueCFGOnlyPipeline:
    def __call__(self, prompt, true_cfg_scale=1):
        raise NotImplementedError


class GuidanceOnlyPipeline:
    def __call__(self, prompt, guidance_scale=1):
        raise NotImplementedError


@pytest.mark.parametrize(
    ("pipeline", "expected"),
    [
        (BothGuidancePipeline(False), "true_cfg_scale"),
        (BothGuidancePipeline(True), "guidance_scale"),
        (TrueCFGOnlyPipeline(), "true_cfg_scale"),
        (GuidanceOnlyPipeline(), "guidance_scale"),
    ],
)
def test_guidance_scale_maps_by_pipeline_capability(pipeline, expected):
    call_args = {}

    add_guidance_call_arg(call_args, pipeline, 4)

    assert call_args == {expected: 4}


def test_ambiguous_guidance_capability_is_rejected():
    pipeline = BothGuidancePipeline(None)

    with pytest.raises(TaskInvalid):
        add_guidance_call_arg({}, pipeline, 4)


def test_true_cfg_injects_empty_negative_prompt_when_omitted():
    pipeline = BothGuidancePipeline(False)
    args = make_args(task_config={"guidance_scale": 4})

    call_args = pipeline_module.build_pipeline_call_args(pipeline, args)

    assert call_args["true_cfg_scale"] == 4
    assert call_args["negative_prompt"] == ""


class MultiImagePipeline:
    def __call__(
        self,
        prompt,
        image: Image.Image | list[Image.Image] | None = None,
        num_images_per_prompt=1,
        **kwargs,
    ):
        raise NotImplementedError


class StrengthPipeline:
    def __call__(
        self,
        prompt,
        image: Image.Image,
        strength=0.8,
        num_images_per_prompt=1,
    ):
        raise NotImplementedError


class MinimalPipeline:
    def __call__(self, prompt, num_images_per_prompt=1):
        raise NotImplementedError


def test_multi_reference_images_are_passed_when_supported():
    args = make_args(
        images=[image_data_url(), image_data_url()],
        task_config={"num_images_per_prompt": 2},
    )

    call_args = pipeline_module.build_pipeline_call_args(MultiImagePipeline(), args)

    assert len(call_args["image"]) == 2
    assert call_args["num_images_per_prompt"] == 2


def test_strength_is_passed_when_supported():
    args = make_args(images=[image_data_url()], strength=0.5)

    call_args = pipeline_module.build_pipeline_call_args(StrengthPipeline(), args)

    assert call_args["strength"] == 0.5


def test_img2img_batch_list_is_not_treated_as_reference_images():
    args = make_args(images=[image_data_url(), image_data_url()])

    with pytest.raises(TaskInvalid):
        pipeline_module.build_pipeline_call_args(StrengthPipeline(), args)


def test_single_image_is_passed_as_one_image():
    args = make_args(images=[image_data_url()])

    call_args = pipeline_module.build_pipeline_call_args(MultiImagePipeline(), args)

    assert isinstance(call_args["image"], Image.Image)


def test_unsupported_kwargs_are_not_passed():
    call_args = pipeline_module.build_pipeline_call_args(MinimalPipeline(), make_args())

    assert set(call_args) == {"prompt", "num_images_per_prompt"}


def test_unsupported_negative_prompt_is_rejected():
    with pytest.raises(TaskInvalid):
        pipeline_module.build_pipeline_call_args(
            MinimalPipeline(), make_args(negative_prompt="low quality")
        )


def test_ineffective_distilled_negative_prompt_is_rejected():
    class DistilledPipeline:
        transformer = SimpleNamespace(config=SimpleNamespace(guidance_embeds=True))

        def __call__(self, prompt, negative_prompt=None, guidance_scale=1):
            raise NotImplementedError

    with pytest.raises(TaskInvalid):
        pipeline_module.build_pipeline_call_args(
            DistilledPipeline(), make_args(negative_prompt="low quality")
        )


def test_true_cfg_negative_prompt_requires_effective_scale():
    with pytest.raises(TaskInvalid):
        pipeline_module.build_pipeline_call_args(
            BothGuidancePipeline(False),
            make_args(negative_prompt="low quality"),
        )

    call_args = pipeline_module.build_pipeline_call_args(
        BothGuidancePipeline(False),
        make_args(
            negative_prompt="low quality",
            task_config={"guidance_scale": 4},
        ),
    )

    assert call_args["negative_prompt"] == "low quality"
    assert call_args["true_cfg_scale"] == 4


def test_unsupported_strength_is_rejected():
    with pytest.raises(TaskInvalid):
        pipeline_module.build_pipeline_call_args(
            MinimalPipeline(), make_args(strength=0.5)
        )


def test_pipeline_class_partitions_cache_key():
    args = make_args()

    native_key = generate_model_key(args, NativeUnifiedPipeline)
    same_native_key = generate_model_key(
        make_args(images=[image_data_url()]), NativeUnifiedPipeline
    )
    image_to_image_key = generate_model_key(args, ImageToImagePipeline)

    assert native_key == same_native_key
    assert native_key != image_to_image_key
