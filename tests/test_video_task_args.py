import pytest
from pydantic import ValidationError

from sd_task.task_args import VideoTaskArgs, derive_video_input_mode


def valid_args() -> dict:
    return {
        "version": "3.8.0",
        "base_model": {"name": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"},
        "prompt": "A cat walking through a sunlit garden",
        "task_config": {},
    }


def test_unknown_top_level_field_is_rejected():
    data = valid_args()
    data["unsupported"] = True

    with pytest.raises(ValidationError):
        VideoTaskArgs.model_validate(data)


def test_unknown_task_config_field_is_rejected():
    data = valid_args()
    data["task_config"] = {"num_videos_per_prompt": 2}

    with pytest.raises(ValidationError):
        VideoTaskArgs.model_validate(data)


def test_empty_prompt_and_media_is_rejected():
    data = {
        "base_model": {"name": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"},
        "prompt": "   ",
    }

    with pytest.raises(ValidationError):
        VideoTaskArgs.model_validate(data)


def test_keyframes_cannot_combine_with_image_or_last_image():
    data = valid_args()
    data.update(
        {
            "image": "data:image/png;base64,AA==",
            "keyframes": [
                {
                    "image": "data:image/png;base64,AA==",
                    "frame_index": 0,
                    "strength": 1.0,
                }
            ],
        }
    )

    with pytest.raises(ValidationError):
        VideoTaskArgs.model_validate(data)


def test_mask_video_requires_video_or_animate():
    data = valid_args()
    data["mask_video"] = "data:video/mp4;base64,AA=="

    with pytest.raises(ValidationError):
        VideoTaskArgs.model_validate(data)


def test_animate_replace_requires_background_and_mask_video():
    data = valid_args()
    data["image"] = "data:image/png;base64,AA=="
    data["animate"] = {
        "mode": "replace",
        "pose_video": "data:video/mp4;base64,AA==",
        "face_video": "data:video/mp4;base64,AA==",
    }

    with pytest.raises(ValidationError):
        VideoTaskArgs.model_validate(data)

    data["animate"]["background_video"] = "data:video/mp4;base64,AA=="
    with pytest.raises(ValidationError):
        VideoTaskArgs.model_validate(data)

    data["mask_video"] = "data:video/mp4;base64,AA=="
    args = VideoTaskArgs.model_validate(data)
    assert derive_video_input_mode(args) == "animate"


def test_audio_requires_visual_conditioning():
    data = valid_args()
    data["audio"] = "data:audio/wav;base64,AA=="

    with pytest.raises(ValidationError):
        VideoTaskArgs.model_validate(data)

    data["image"] = "data:image/png;base64,AA=="
    args = VideoTaskArgs.model_validate(data)
    assert derive_video_input_mode(args) == "a2v"


def test_audio_only_references_are_rejected():
    data = valid_args()
    data["references"] = [
        {"type": "audio", "data": "data:audio/wav;base64,AA=="}
    ]

    with pytest.raises(ValidationError):
        VideoTaskArgs.model_validate(data)


def test_pipeline_args_passthrough_and_reserved_key_rejection():
    data = valid_args()
    data["pipeline_args"] = {
        "guidance_scale_2": 1.0,
        "audio_guidance_scale": 7.0,
        "stg_scale": 1.0,
        "sigmas": [1.0, 0.5, 0.0],
    }

    args = VideoTaskArgs.model_validate(data)
    assert args.pipeline_args["audio_guidance_scale"] == 7.0

    for reserved_key in (
        "generator",
        "latents",
        "width",
        "num_frames",
        "num_videos_per_prompt",
        "enable_prompt_enhancement",
    ):
        bad = valid_args()
        bad["pipeline_args"] = {reserved_key: 1}
        with pytest.raises(ValidationError):
            VideoTaskArgs.model_validate(bad)


def test_multi_lora_quantization_and_components():
    data = valid_args()
    data.update(
        {
            "image": "data:image/png;base64,AA==",
            "last_image": "data:image/png;base64,BB==",
            "task_config": {
                "width": 720,
                "height": 1280,
                "num_frames": 81,
                "fps": 16,
                "steps": 4,
                "seed": 42,
                "guidance_scale": 1.0,
            },
            "loras": [
                {
                    "model": "lightx2v/Wan2.2-Lightning",
                    "weight_file_name": "high.safetensors",
                    "weight": 100,
                    "target": "transformer",
                },
                {
                    "model": "lightx2v/Wan2.2-Lightning",
                    "weight_file_name": "low.safetensors",
                    "weight": 85,
                    "target": "transformer_2",
                },
            ],
            "components": {
                "text_encoder": {
                    "format": "diffusers",
                    "repo": "org/custom-text-encoder",
                    "revision": None,
                    "format_kwargs": {"subfolder": "text_encoder"},
                }
            },
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "components_to_quantize": ["transformer", "transformer_2"],
                "quant_kwargs": {"load_in_4bit": True},
            },
        }
    )

    args = VideoTaskArgs.model_validate(data)
    assert derive_video_input_mode(args) == "flf2v"
    assert args.loras[0].weight == 1.0
    assert args.loras[1].weight == 0.85
    assert args.loras[1].target == "transformer_2"
    assert args.quantization_config.components_to_quantize == [
        "transformer",
        "transformer_2",
    ]


@pytest.mark.parametrize(
    ("extra", "expected_mode"),
    [
        ({}, "t2v"),
        ({"image": "data:image/png;base64,AA=="}, "i2v"),
        (
            {
                "image": "data:image/png;base64,AA==",
                "last_image": "data:image/png;base64,BB==",
            },
            "flf2v",
        ),
        (
            {
                "keyframes": [
                    {"image": "data:image/png;base64,AA==", "frame_index": 0}
                ]
            },
            "keyframes",
        ),
        (
            {
                "references": [
                    {"type": "image", "data": "data:image/png;base64,AA=="}
                ]
            },
            "r2v",
        ),
        ({"video": "data:video/mp4;base64,AA=="}, "v2v"),
    ],
)
def test_derive_video_input_mode_precedence(extra: dict, expected_mode: str):
    data = valid_args()
    data.update(extra)
    args = VideoTaskArgs.model_validate(data)
    assert derive_video_input_mode(args) == expected_mode
