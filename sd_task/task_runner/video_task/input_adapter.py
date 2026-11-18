from __future__ import annotations

import copy
import inspect
from typing import Any

import torch
from diffusers import ModularPipeline

from sd_task.task_args.video_task import VideoTaskArgs
from sd_task.task_runner.image_task.errors import TaskInvalid
from sd_task.task_runner.image_task.guidance import (
    add_guidance_call_arg,
    supports_negative_prompt_guidance,
)

from .media import decode_audio_input, decode_image_input, decode_video_input


def resolve_spatial_multiple(pipeline: Any) -> int:
    canvas_multiple = getattr(pipeline, "canvas_multiple", None)
    if isinstance(canvas_multiple, int) and canvas_multiple > 0:
        return canvas_multiple

    vae_scale = getattr(pipeline, "vae_scale_factor_spatial", None)
    if not isinstance(vae_scale, int) or vae_scale <= 0:
        vae_scale = getattr(pipeline, "vae_spatial_compression_ratio", None)
    if not isinstance(vae_scale, int) or vae_scale <= 0:
        vae = getattr(pipeline, "vae", None)
        vae_scale = getattr(vae, "spatial_compression_ratio", None)
    if not isinstance(vae_scale, int) or vae_scale <= 0:
        vae_scale = 8

    patch_spatial = 1
    for attr in ("transformer", "transformer_2", "transformer_ref"):
        denoiser = getattr(pipeline, attr, None)
        cfg = getattr(denoiser, "config", None)
        patch = getattr(cfg, "patch_size", None)
        if isinstance(patch, int) and patch > 0:
            patch_spatial = patch
            break
        if isinstance(patch, (list, tuple)) and len(patch) > 0:
            last = patch[-1]
            if isinstance(last, int) and last > 0:
                patch_spatial = last
                break

    return vae_scale * patch_spatial


def validate_geometry_and_temporal_grid(pipeline: Any, args: VideoTaskArgs) -> None:
    multiple = resolve_spatial_multiple(pipeline)
    if args.task_config.width is not None and args.task_config.width % multiple != 0:
        raise TaskInvalid(
            f"task_config.width ({args.task_config.width}) must be divisible by {multiple}"
        )
    if args.task_config.height is not None and args.task_config.height % multiple != 0:
        raise TaskInvalid(
            f"task_config.height ({args.task_config.height}) must be divisible by {multiple}"
        )

    num_frames = args.task_config.num_frames
    frames_per_chunk = getattr(pipeline, "vae_frames_per_chunk", None)
    latents_per_chunk = getattr(pipeline, "vae_latents_per_chunk", None)
    if (
        isinstance(frames_per_chunk, int)
        and frames_per_chunk > 0
        and isinstance(latents_per_chunk, int)
        and latents_per_chunk > 0
    ):
        if num_frames % frames_per_chunk != latents_per_chunk:
            raise TaskInvalid(
                f"task_config.num_frames ({num_frames}) must satisfy "
                f"num_frames % {frames_per_chunk} == {latents_per_chunk}"
            )
        min_dur = getattr(pipeline, "min_duration", None)
        max_dur = getattr(pipeline, "max_duration", None)
        pipe_fps = float(getattr(pipeline, "fps", 24.0))
        if isinstance(min_dur, (int, float)) and num_frames / pipe_fps < float(min_dur) - 1e-6:
            raise TaskInvalid(
                f"task_config.num_frames ({num_frames}) is shorter than pipeline minimum duration ({min_dur}s)"
            )
        if isinstance(max_dur, (int, float)) and num_frames / pipe_fps > float(max_dur) + 1e-6:
            raise TaskInvalid(
                f"task_config.num_frames ({num_frames}) exceeds pipeline maximum duration ({max_dur}s)"
            )
        return

    temporal_ratio = getattr(pipeline, "vae_scale_factor_temporal", None)
    if not isinstance(temporal_ratio, int) or temporal_ratio <= 0:
        temporal_ratio = getattr(pipeline, "vae_temporal_compression_ratio", None)
    if not isinstance(temporal_ratio, int) or temporal_ratio <= 0:
        vae = getattr(pipeline, "vae", None)
        temporal_ratio = getattr(vae, "temporal_compression_ratio", None)

    if isinstance(temporal_ratio, int) and temporal_ratio > 1 and args.animate is None:
        if (num_frames - 1) % temporal_ratio != 0:
            raise TaskInvalid(
                f"task_config.num_frames ({num_frames}) must satisfy "
                f"(num_frames - 1) % {temporal_ratio} == 0"
            )


def resolve_effective_fps(pipeline: Any, args: VideoTaskArgs) -> float:
    if args.task_config.fps is not None:
        return float(args.task_config.fps)
    pipe_fps = getattr(pipeline, "fps", None)
    if isinstance(pipe_fps, (int, float)) and float(pipe_fps) > 0:
        return float(pipe_fps)
    if not isinstance(pipeline, ModularPipeline):
        params = inspect.signature(pipeline.__call__).parameters
        for key in ("frame_rate", "fps"):
            param = params.get(key)
            if (
                param is not None
                and param.default is not inspect.Parameter.empty
                and isinstance(param.default, (int, float))
                and float(param.default) > 0
            ):
                return float(param.default)
    return 16.0


def _build_condition_object(pipeline: Any, frames: Any, index: int, strength: float) -> Any:
    module_name = type(pipeline).__module__
    if "ltx2" in module_name:
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

        return LTX2VideoCondition(frames=frames, index=index, strength=strength)
    if "ltx" in module_name:
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        return LTXVideoCondition(frames=frames, index=index, strength=strength)
    raise TaskInvalid("The selected pipeline does not expose a supported condition container")


def _adapt_references(
    pipeline: Any,
    accepted_names: set[str],
    args: VideoTaskArgs,
    call_args: dict[str, Any],
) -> None:
    if not args.references:
        return

    module_name = type(pipeline).__module__
    if "minimax_h3" in module_name and "references" in accepted_names:
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3AudioReference,
            MiniMaxH3ImageReference,
            MiniMaxH3VideoReference,
        )

        img_count = sum(1 for r in args.references if r.type == "image")
        vid_count = sum(1 for r in args.references if r.type == "video")
        aud_count = sum(1 for r in args.references if r.type == "audio")
        if (
            img_count > 9
            or vid_count > 3
            or aud_count > 3
            or len(args.references) > 12
        ):
            raise TaskInvalid(
                "References exceed MiniMax-H3 limits (<=9 images, <=3 videos, <=3 audios, <=12 total)"
            )

        converted: list[Any] = []
        for item in args.references:
            if item.type == "image":
                converted.append(
                    MiniMaxH3ImageReference(image=decode_image_input(item.data))
                )
            elif item.type == "video":
                vid = decode_video_input(item.data)
                converted.append(
                    MiniMaxH3VideoReference(
                        frames=vid.frames,
                        fps=vid.fps,
                        audio=vid.audio,
                        sample_rate=vid.audio_sample_rate,
                    )
                )
            elif item.type == "audio":
                aud = decode_audio_input(item.data)
                converted.append(
                    MiniMaxH3AudioReference(
                        audio=aud.waveform, sample_rate=aud.sample_rate
                    )
                )
        call_args["references"] = converted
        return

    if "reference_images" in accepted_names:
        if any(item.type != "image" for item in args.references):
            raise TaskInvalid(
                "The selected pipeline accepts only image items in references"
            )
        call_args["reference_images"] = [
            decode_image_input(item.data) for item in args.references
        ]
        return

    if "reference_conditions" in accepted_names:
        conds = []
        for item in args.references:
            if item.type == "image":
                conds.append(
                    _build_condition_object(
                        pipeline, decode_image_input(item.data), 0, 1.0
                    )
                )
            elif item.type == "video":
                vid = decode_video_input(item.data)
                conds.append(_build_condition_object(pipeline, vid.frames, 0, 1.0))
            else:
                raise TaskInvalid(
                    "The selected pipeline does not support audio reference_conditions"
                )
        call_args["reference_conditions"] = conds
        return

    if (
        "image" in accepted_names
        and "strength" not in accepted_names
        and args.image is None
        and all(item.type == "image" for item in args.references)
    ):
        imgs = [decode_image_input(item.data) for item in args.references]
        call_args["image"] = imgs[0] if len(imgs) == 1 else imgs
        return

    raise TaskInvalid("The selected pipeline does not support references")


def build_video_call_args(
    pipeline: Any,
    args: VideoTaskArgs,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build validated call kwargs and temporary guider overrides (restored after call)."""
    validate_geometry_and_temporal_grid(pipeline, args)
    effective_fps = resolve_effective_fps(pipeline, args)
    generator = torch.Generator(device="cpu").manual_seed(args.task_config.seed)

    is_modular = isinstance(pipeline, ModularPipeline)
    if is_modular:
        accepted_params: dict[str, Any] = {
            p.name: p
            for p in getattr(pipeline.blocks, "inputs", [])
            if getattr(p, "name", None)
        }
    else:
        accepted_params = dict(inspect.signature(pipeline.__call__).parameters)

    accepted_names = set(accepted_params)
    call_args: dict[str, Any] = {}
    guider_restore: dict[str, Any] = {}

    if args.prompt is not None:
        if "prompt" not in accepted_names:
            raise TaskInvalid("The selected video pipeline does not accept prompt")
        call_args["prompt"] = args.prompt

    if args.task_config.width is not None:
        if "width" not in accepted_names:
            raise TaskInvalid("The selected video pipeline does not accept width")
        call_args["width"] = args.task_config.width

    if args.task_config.height is not None:
        if "height" not in accepted_names:
            raise TaskInvalid("The selected video pipeline does not accept height")
        call_args["height"] = args.task_config.height

    if "segment_frame_length" in accepted_names and "num_frames" not in accepted_names:
        call_args["segment_frame_length"] = args.task_config.num_frames
    elif "num_frames" in accepted_names:
        call_args["num_frames"] = args.task_config.num_frames

    if "frame_rate" in accepted_names:
        call_args["frame_rate"] = float(effective_fps)
    elif "fps" in accepted_names:
        call_args["fps"] = float(effective_fps)

    if "num_inference_steps" in accepted_names:
        call_args["num_inference_steps"] = args.task_config.steps

    if "generator" in accepted_names:
        call_args["generator"] = generator

    if "num_videos_per_prompt" in accepted_names:
        call_args["num_videos_per_prompt"] = 1

    if "enable_prompt_enhancement" in accepted_names:
        call_args["enable_prompt_enhancement"] = False

    if "output_type" in accepted_names:
        call_args["output_type"] = "np"

    # Guidance & negative prompt handling
    if is_modular:
        guider = getattr(pipeline, "guider", None)
        if args.task_config.guidance_scale is not None:
            if "guidance_scale" in accepted_names:
                call_args["guidance_scale"] = args.task_config.guidance_scale
            elif guider is not None and hasattr(guider, "guidance_scale"):
                guider_restore["guider"] = guider
                cloned = copy.deepcopy(guider)
                cloned.guidance_scale = args.task_config.guidance_scale
                pipeline.update_components(guider=cloned)
            else:
                raise TaskInvalid(
                    "The selected modular pipeline does not support guidance_scale"
                )
        if args.negative_prompt:
            if "negative_prompt" not in accepted_names:
                raise TaskInvalid(
                    "The selected modular pipeline does not support negative_prompt"
                )
            eff_scale = args.task_config.guidance_scale
            if eff_scale is None and guider is not None:
                eff_scale = getattr(guider, "guidance_scale", None)
            if eff_scale is not None and float(eff_scale) <= 1.0:
                raise TaskInvalid(
                    "negative_prompt requires effective guidance_scale > 1.0"
                )
            call_args["negative_prompt"] = args.negative_prompt
        elif args.negative_prompt is not None and "negative_prompt" in accepted_names:
            call_args["negative_prompt"] = args.negative_prompt
    else:
        if args.negative_prompt:
            if not supports_negative_prompt_guidance(
                pipeline, accepted_params, args.task_config.guidance_scale
            ):
                raise TaskInvalid(
                    "The selected video pipeline does not support effective negative_prompt guidance"
                )
            call_args["negative_prompt"] = args.negative_prompt
        elif args.negative_prompt is not None and "negative_prompt" in accepted_names:
            call_args["negative_prompt"] = args.negative_prompt

        guidance_param = add_guidance_call_arg(
            call_args, pipeline, args.task_config.guidance_scale
        )
        if (
            guidance_param == "true_cfg_scale"
            and "negative_prompt" in accepted_names
            and "negative_prompt" not in call_args
        ):
            call_args["negative_prompt"] = ""

    # Frame / keyframe conditioning
    if args.keyframes:
        if "conditions" not in accepted_names:
            raise TaskInvalid("The selected video pipeline does not support keyframes")
        call_args["conditions"] = [
            _build_condition_object(
                pipeline,
                decode_image_input(kf.image),
                kf.frame_index,
                kf.strength,
            )
            for kf in args.keyframes
        ]
    elif args.image is not None and args.last_image is not None:
        first_img = decode_image_input(args.image)
        last_img = decode_image_input(args.last_image)
        if "image" in accepted_names and "last_image" in accepted_names:
            call_args["image"] = first_img
            call_args["last_image"] = last_img
        elif "conditions" in accepted_names:
            call_args["conditions"] = [
                _build_condition_object(pipeline, first_img, 0, 1.0),
                _build_condition_object(pipeline, last_img, -1, 1.0),
            ]
        else:
            raise TaskInvalid(
                "The selected video pipeline does not support first+last frame conditioning"
            )
    elif args.last_image is not None:
        last_img = decode_image_input(args.last_image)
        if "last_image" in accepted_names:
            call_args["last_image"] = last_img
        elif "conditions" in accepted_names:
            call_args["conditions"] = [
                _build_condition_object(pipeline, last_img, -1, 1.0)
            ]
        else:
            raise TaskInvalid("The selected video pipeline does not support last_image")
    elif args.image is not None:
        first_img = decode_image_input(args.image)
        if "image" in accepted_names:
            call_args["image"] = first_img
        elif "conditions" in accepted_names:
            call_args["conditions"] = [
                _build_condition_object(pipeline, first_img, 0, 1.0)
            ]
        else:
            raise TaskInvalid("The selected video pipeline does not support image input")

    _adapt_references(pipeline, accepted_names, args, call_args)

    # Video / mask / strength / conditioning_scale / animate / audio
    if args.video is not None:
        decoded_vid = decode_video_input(args.video)
        if "driving_video" in accepted_names:
            call_args["driving_video"] = decoded_vid.frames
            if "driving_video_fps" in accepted_names:
                call_args["driving_video_fps"] = decoded_vid.fps
        elif "video" in accepted_names:
            call_args["video"] = decoded_vid.frames
        else:
            raise TaskInvalid("The selected video pipeline does not support video input")

    if args.mask_video is not None:
        decoded_mask = decode_video_input(args.mask_video)
        if "mask_video" in accepted_names:
            call_args["mask_video"] = decoded_mask.frames
        elif "mask" in accepted_names:
            call_args["mask"] = decoded_mask.frames
        else:
            raise TaskInvalid("The selected video pipeline does not support mask_video")

    if args.strength is not None:
        if "strength" not in accepted_names:
            raise TaskInvalid("The selected video pipeline does not support strength")
        call_args["strength"] = args.strength

    if args.conditioning_scale is not None:
        if "conditioning_scale" not in accepted_names:
            raise TaskInvalid(
                "The selected video pipeline does not support conditioning_scale"
            )
        call_args["conditioning_scale"] = args.conditioning_scale

    if args.audio is not None:
        decoded_aud = decode_audio_input(args.audio)
        if "audio" in accepted_names:
            call_args["audio"] = decoded_aud.waveform
            if "audio_sample_rate" in accepted_names:
                call_args["audio_sample_rate"] = decoded_aud.sample_rate
            elif "sampling_rate" in accepted_names:
                call_args["sampling_rate"] = decoded_aud.sample_rate
        elif "minimax_h3" in type(pipeline).__module__ and "references" in accepted_names and not args.references and args.image is not None:
            from diffusers.modular_pipelines.minimax_h3 import (
                MiniMaxH3AudioReference,
                MiniMaxH3ImageReference,
            )

            first_img = call_args.pop("image")
            call_args["references"] = [
                MiniMaxH3ImageReference(image=first_img),
                MiniMaxH3AudioReference(
                    audio=decoded_aud.waveform,
                    sample_rate=decoded_aud.sample_rate,
                ),
            ]
        else:
            raise TaskInvalid("The selected video pipeline does not support audio input")

    if args.animate is not None:
        if args.animate.pose_video is not None:
            if "pose_video" not in accepted_names:
                raise TaskInvalid("The selected pipeline does not support animate.pose_video")
            call_args["pose_video"] = decode_video_input(args.animate.pose_video).frames
        if args.animate.face_video is not None:
            if "face_video" not in accepted_names:
                raise TaskInvalid("The selected pipeline does not support animate.face_video")
            call_args["face_video"] = decode_video_input(args.animate.face_video).frames
        if args.animate.background_video is not None:
            if "background_video" not in accepted_names:
                raise TaskInvalid(
                    "The selected pipeline does not support animate.background_video"
                )
            call_args["background_video"] = decode_video_input(
                args.animate.background_video
            ).frames
        if "mode" in accepted_names:
            call_args["mode"] = args.animate.mode
        if args.animate.prev_segment_conditioning_frames is not None:
            if "prev_segment_conditioning_frames" not in accepted_names:
                raise TaskInvalid(
                    "The selected pipeline does not support animate.prev_segment_conditioning_frames"
                )
            call_args["prev_segment_conditioning_frames"] = (
                args.animate.prev_segment_conditioning_frames
            )

    # Passthrough pipeline_args validation & routing
    if args.pipeline_args:
        for key, value in args.pipeline_args.items():
            if key in accepted_names:
                call_args[key] = value
                continue
            if is_modular:
                matched_guider = False
                for g_name in ("guider", "guider_2", "audio_guider"):
                    g_obj = getattr(pipeline, g_name, None)
                    if g_obj is not None and hasattr(g_obj, key):
                        if g_name not in guider_restore:
                            guider_restore[g_name] = g_obj
                            g_obj = copy.deepcopy(g_obj)
                            pipeline.update_components(**{g_name: g_obj})
                        setattr(getattr(pipeline, g_name), key, value)
                        matched_guider = True
                        break
                if matched_guider:
                    continue
            raise TaskInvalid(
                f"pipeline_args key '{key}' is not supported by the resolved pipeline"
            )

    return call_args, guider_restore
