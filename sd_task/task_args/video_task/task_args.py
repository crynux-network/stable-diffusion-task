from typing import Any, Literal

from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated

from ..image_task.task_args import (
    BaseModelArgs,
    ComponentSource,
    QuantizationConfig,
)
from ..types import FloatFractionAsInt, NonEmptyString

VideoInputMode = Literal[
    "t2v",
    "i2v",
    "flf2v",
    "keyframes",
    "r2v",
    "v2v",
    "animate",
    "a2v",
]

RESERVED_PIPELINE_ARG_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "prompt_2",
        "negative_prompt",
        "negative_prompt_2",
        "image",
        "last_image",
        "video",
        "driving_video",
        "mask",
        "mask_video",
        "pose_video",
        "face_video",
        "background_video",
        "conditions",
        "reference_conditions",
        "references",
        "reference_images",
        "width",
        "height",
        "num_frames",
        "segment_frame_length",
        "min_seconds",
        "max_seconds",
        "fps",
        "frame_rate",
        "driving_video_fps",
        "num_inference_steps",
        "guidance_scale",
        "true_cfg_scale",
        "strength",
        "conditioning_scale",
        "num_videos_per_prompt",
        "generator",
        "latents",
        "audio_latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "prompt_embeds_2",
        "negative_prompt_embeds_2",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
        "prompt_attention_mask",
        "negative_prompt_attention_mask",
        "image_embeds",
        "output_type",
        "return_dict",
        "callback_on_step_end",
        "callback_on_step_end_tensor_inputs",
        "enable_prompt_enhancement",
        "system_prompt",
        "prompt_max_new_tokens",
        "prompt_enhancement_kwargs",
        "prompt_enhancement_seed",
        "attention_kwargs",
    }
)


def _ensure_json_compatible(value: Any, path: str) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json_compatible(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _ensure_json_compatible(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} must contain JSON-compatible values")


class VideoLoraArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: NonEmptyString
    revision: NonEmptyString | None = None
    weight_file_name: str = ""
    weight: FloatFractionAsInt = 100
    target: Literal[
        "auto",
        "transformer",
        "transformer_2",
        "transformer_ref",
        "both",
    ] = "auto"


class KeyframeCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: NonEmptyString
    frame_index: int
    strength: Annotated[float, Ge(0.0), Le(1.0)] = 1.0


class MediaReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image", "video", "audio"]
    data: NonEmptyString


class AnimateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["animate", "replace"] = "animate"
    pose_video: NonEmptyString | None = None
    face_video: NonEmptyString | None = None
    background_video: NonEmptyString | None = None
    prev_segment_conditioning_frames: Annotated[int, Gt(0), Le(16)] | None = None

    @model_validator(mode="after")
    def _validate_replace_inputs(self) -> "AnimateArgs":
        if self.mode == "replace" and self.background_video is None:
            raise ValueError(
                "animate.background_video is required when animate.mode is 'replace'"
            )
        return self


class VideoTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: Annotated[int, Gt(0), Le(4096)] | None = None
    height: Annotated[int, Gt(0), Le(4096)] | None = None
    num_frames: Annotated[int, Gt(0), Le(1025)] = 81
    fps: Annotated[float, Gt(0.0), Le(120.0)] | None = None
    steps: Annotated[int, Gt(0), Le(100)] = 30
    seed: Annotated[int, Ge(0), Lt(2147483648)] = 0
    guidance_scale: Annotated[float, Ge(0.0)] | None = None


class VideoTaskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: NonEmptyString
    base_model: BaseModelArgs
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto"

    prompt: str | None = None
    negative_prompt: str | None = None

    image: NonEmptyString | None = None
    last_image: NonEmptyString | None = None
    keyframes: list[KeyframeCondition] = Field(default_factory=list)
    references: list[MediaReference] = Field(default_factory=list)

    video: NonEmptyString | None = None
    mask_video: NonEmptyString | None = None
    audio: NonEmptyString | None = None

    strength: Annotated[float, Ge(0.0), Le(1.0)] | None = None
    conditioning_scale: Annotated[float, Ge(0.0), Le(2.0)] | None = None
    animate: AnimateArgs | None = None

    task_config: VideoTaskConfig = Field(default_factory=VideoTaskConfig)
    pipeline_args: dict[str, Any] | None = None

    loras: list[VideoLoraArgs] = Field(default_factory=list)
    components: dict[str, ComponentSource] = Field(default_factory=dict)
    quantization_config: QuantizationConfig | None = None

    @model_validator(mode="after")
    def _validate_video_args(self) -> "VideoTaskArgs":
        has_prompt = self.prompt is not None and len(self.prompt.strip()) > 0
        has_media = any(
            (
                self.image is not None,
                self.last_image is not None,
                bool(self.keyframes),
                bool(self.references),
                self.video is not None,
                self.animate is not None,
            )
        )
        if not has_prompt and not has_media:
            raise ValueError(
                "At least one of prompt or conditioning media must be provided"
            )

        if self.keyframes and (
            self.image is not None or self.last_image is not None
        ):
            raise ValueError(
                "keyframes must not be combined with image or last_image"
            )

        if (
            self.mask_video is not None
            and self.video is None
            and self.animate is None
        ):
            raise ValueError("mask_video requires video or animate")

        if (
            self.animate is not None
            and self.animate.mode == "replace"
            and self.mask_video is None
        ):
            raise ValueError(
                "mask_video is required when animate.mode is 'replace'"
            )

        if (
            self.audio is not None
            and self.image is None
            and not self.references
        ):
            raise ValueError("audio conditioning requires image or references")

        if self.references and all(
            item.type == "audio" for item in self.references
        ):
            raise ValueError(
                "references must not contain only audio items without visual conditioning"
            )

        if self.pipeline_args is not None:
            _ensure_json_compatible(self.pipeline_args, "pipeline_args")
            reserved = sorted(
                set(self.pipeline_args).intersection(RESERVED_PIPELINE_ARG_KEYS)
            )
            if reserved:
                joined = ", ".join(reserved)
                raise ValueError(
                    f"pipeline_args must not contain reserved keys: {joined}"
                )

        return self


def derive_video_input_mode(args: VideoTaskArgs) -> VideoInputMode:
    if args.animate is not None:
        return "animate"
    if args.video is not None:
        return "v2v"
    if args.references:
        return "r2v"
    if args.keyframes:
        return "keyframes"
    if args.last_image is not None:
        return "flf2v"
    if args.audio is not None:
        return "a2v"
    if args.image is not None:
        return "i2v"
    return "t2v"
