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
