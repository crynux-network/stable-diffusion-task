import pytest
from pydantic import ValidationError

from sd_task.task_args import ImageTaskArgs


def valid_args() -> dict:
    return {
        "version": "3.7.0",
        "base_model": {"name": "test/model"},
        "prompt": "test",
        "task_config": {},
    }


def test_unknown_top_level_field_is_rejected():
    data = valid_args()
    data["unsupported"] = True

    with pytest.raises(ValidationError):
        ImageTaskArgs.model_validate(data)


def test_unknown_base_model_field_is_rejected():
    data = valid_args()
    data["base_model"]["unsupported"] = True

    with pytest.raises(ValidationError):
        ImageTaskArgs.model_validate(data)


def test_unknown_task_config_field_is_rejected():
    data = valid_args()
    data["task_config"] = {"unsupported": True}

    with pytest.raises(ValidationError):
        ImageTaskArgs.model_validate(data)


def test_image_task_accepts_generation_and_editing_fields():
    data = valid_args()
    data.update(
        {
            "negative_prompt": "low quality",
            "images": ["data:image/png;base64,AA=="],
            "strength": 0.7,
            "task_config": {
                "num_images_per_prompt": 2,
                "guidance_scale": 4,
            },
        }
    )

    args = ImageTaskArgs.model_validate(data)

    assert args.task_config.num_images_per_prompt == 2
    assert args.task_config.guidance_scale == 4
    assert args.strength == 0.7


def test_image_task_accepts_quantization_and_components():
    data = valid_args()
    data.update(
        {
            "base_model": {"name": "test/model", "revision": None},
            "components": {
                "transformer": {
                    "format": "gguf",
                    "repo": "org/quantized",
                    "revision": None,
                    "format_kwargs": {"filename": "transformer-Q4_K_M.gguf"},
                }
            },
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "components_to_quantize": ["text_encoder"],
                "quant_kwargs": {"load_in_8bit": True},
            },
        }
    )

    args = ImageTaskArgs.model_validate(data)
    assert args.components["transformer"].format == "gguf"
    assert args.quantization_config.quant_method == "bitsandbytes"
    assert args.quantization_config.components_to_quantize == ["text_encoder"]


def test_image_task_accepts_diffusers_components():
    data = valid_args()
    data["components"] = {
        "vae": {
            "format": "diffusers",
            "repo": "org/vae",
            "revision": None,
            "format_kwargs": {"subfolder": "vae"},
        }
    }
    args = ImageTaskArgs.model_validate(data)
    assert args.components["vae"].format == "diffusers"
    assert args.components["vae"].format_kwargs["subfolder"] == "vae"


def test_quantization_rejects_duplicate_components():
    data = valid_args()
    data["quantization_config"] = {
        "quant_method": "bitsandbytes",
        "components_to_quantize": ["transformer", "transformer"],
        "quant_kwargs": {"load_in_4bit": True},
    }
    with pytest.raises(ValidationError):
        ImageTaskArgs.model_validate(data)


def test_quantization_rejects_dequantize_true():
    data = valid_args()
    data["quantization_config"] = {
        "quant_method": "bitsandbytes",
        "components_to_quantize": ["transformer"],
        "quant_kwargs": {"dequantize": True, "load_in_4bit": True},
    }
    with pytest.raises(ValidationError):
        ImageTaskArgs.model_validate(data)


def test_quantization_rejects_dequantize_false():
    data = valid_args()
    data["quantization_config"] = {
        "quant_method": "bitsandbytes",
        "components_to_quantize": ["transformer"],
        "quant_kwargs": {"dequantize": False, "load_in_4bit": True},
    }
    with pytest.raises(ValidationError):
        ImageTaskArgs.model_validate(data)
