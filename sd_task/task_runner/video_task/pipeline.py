from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Type

import torch
from diffusers import DiffusionPipeline, ModularPipeline
from diffusers.pipelines.auto_pipeline import (
    AUTO_IMAGE2VIDEO_PIPELINES_MAPPING,
    _get_task_class,
)

from sd_task import utils
from sd_task.task_args.video_task import (
    VideoInputMode,
    VideoLoraArgs,
    VideoTaskArgs,
    derive_video_input_mode,
)
from sd_task.task_runner.image_task.errors import TaskInvalid
from sd_task.task_runner.image_task.quantization.diffusers_component import (
    DIFFUSERS_COMPONENT_FORMAT,
    load_diffusers_component,
)
from sd_task.task_runner.image_task.quantization.gguf import (
    load_gguf_transformer,
    pipeline_config_from_base,
)
from sd_task.task_runner.image_task.quantization.registry import (
    resolve_component_loader,
)

SUPPORTED_VIDEO_GGUF_COMPONENTS = frozenset(
    {"transformer", "transformer_2", "transformer_ref"}
)


@dataclass(frozen=True)
class ResolvedVideoPipelineSpec:
    kind: str  # "diffusion" | "modular"
    pipeline_class: Type[Any]
    workflow: str | None = None


def _sibling_diffusion_candidates(class_name: str, input_mode: VideoInputMode) -> list[str]:
    if not class_name.endswith("Pipeline"):
        return []
    prefix = class_name[: -len("Pipeline")]
    for suffix in ("TextToVideo", "T2V", "ImageToVideo", "I2V"):
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]
            break

    if input_mode == "i2v":
        return [
            f"{prefix}ImageToVideoPipeline",
            f"{prefix}I2VPipeline",
            f"{prefix}ConditionPipeline",
        ]
    if input_mode == "flf2v":
        return [
            f"{prefix}ImageToVideoPipeline",
            f"{prefix}ConditionPipeline",
            f"{prefix}FLF2VPipeline",
        ]
    if input_mode == "keyframes":
        return [f"{prefix}ConditionPipeline"]
    if input_mode == "v2v":
        return [
            f"{prefix}VideoToVideoPipeline",
            f"{prefix}VACEPipeline",
            f"{prefix}ConditionPipeline",
        ]
    if input_mode == "r2v":
        return [
            f"{prefix}VACEPipeline",
            f"{prefix}ConditionPipeline",
            f"{prefix}ImageToVideoPipeline",
        ]
    if input_mode == "animate":
        return [f"{prefix}AnimatePipeline"]
    if input_mode == "a2v":
        return [
            f"{prefix}ConditionPipeline",
            f"{prefix}ImageToVideoPipeline",
        ]
    return []


def _diffusion_class_supports_mode(
    pipeline_class: Type[DiffusionPipeline], args: VideoTaskArgs
) -> bool:
    params = set(inspect.signature(pipeline_class.__call__).parameters)
    if args.animate is not None:
        return {"image", "pose_video", "face_video"}.issubset(params) or (
            "image" in params and ("video" in params or "driving_video" in params)
        )
    if args.video is not None:
        return "video" in params or "driving_video" in params or "conditions" in params
    if args.keyframes:
        return "conditions" in params
    if args.references:
        return (
            "reference_images" in params
            or "references" in params
            or "reference_conditions" in params
            or "image" in params
        )
    if args.last_image is not None:
        return "last_image" in params or "conditions" in params
    if args.image is not None:
        return "image" in params or "conditions" in params
    return True


def _resolve_modular_workflow(blocks: Any, args: VideoTaskArgs) -> str | None:
    workflow_map = getattr(blocks, "_workflow_map", None)
    if not workflow_map:
        return None

    block_input_names = {
        param.name for param in getattr(blocks, "inputs", []) if getattr(param, "name", None)
    }

    active_triggers: set[str] = {"num_frames"}
    if args.prompt is not None and args.prompt.strip():
        active_triggers.add("prompt")

    if args.references:
        if "references" in block_input_names:
            active_triggers.add("references")
        elif "reference_conditions" in block_input_names:
            active_triggers.add("reference_conditions")
        else:
            raise TaskInvalid("The selected modular pipeline does not support references")

    if args.keyframes:
        if "conditions" in block_input_names:
            active_triggers.add("conditions")
        else:
            raise TaskInvalid("The selected modular pipeline does not support keyframes")

    if args.image is not None and args.last_image is not None:
        if "last_image" in block_input_names:
            active_triggers.update({"image", "last_image"})
        elif "conditions" in block_input_names:
            active_triggers.add("conditions")
        else:
            raise TaskInvalid("The selected modular pipeline does not support last_image")
    elif args.last_image is not None:
        if "last_image" in block_input_names:
            active_triggers.add("last_image")
        elif "conditions" in block_input_names:
            active_triggers.add("conditions")
        else:
            raise TaskInvalid("The selected modular pipeline does not support last_image")
    elif args.image is not None:
        if "image" in block_input_names:
            active_triggers.add("image")
        elif "conditions" in block_input_names:
            active_triggers.add("conditions")
        else:
            raise TaskInvalid("The selected modular pipeline does not support image input")

    if args.video is not None:
        if "video" in block_input_names:
            active_triggers.add("video")
        elif "driving_video" in block_input_names:
            active_triggers.add("driving_video")
        elif "conditions" in block_input_names:
            active_triggers.add("conditions")
        else:
            raise TaskInvalid("The selected modular pipeline does not support video input")

    non_prompt_active = active_triggers - {"prompt", "num_frames"}
    best_workflow: str | None = None
    best_score = -1

    for wf_name, trigger_spec in workflow_map.items():
        patterns = trigger_spec if isinstance(trigger_spec, tuple) else (trigger_spec,)
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            required_keys = {k for k, v in pattern.items() if v}
            if not required_keys.issubset(active_triggers):
                continue
            non_prompt_required = required_keys - {"prompt", "num_frames"}
            if non_prompt_active and not non_prompt_required:
                continue
            score = len(non_prompt_required) * 10 + len(required_keys)
            if score > best_score:
                best_score = score
                best_workflow = wf_name

    if best_workflow is None:
        raise TaskInvalid(
            "No workflow in the modular pipeline supports the supplied conditioning inputs"
        )
    return best_workflow


def resolve_video_pipeline_spec(
    args: VideoTaskArgs,
    *,
    cache_dir: str,
    local_files_only: bool,
) -> ResolvedVideoPipelineSpec:
    load_args: dict[str, Any] = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if args.base_model.revision is not None:
        load_args["revision"] = args.base_model.revision

    diffusers_mod = __import__("diffusers")
    input_mode = derive_video_input_mode(args)

    modular_config, classic_config = ModularPipeline._load_pipeline_config(
        args.base_model.name, **load_args
    )

    if modular_config is not None and classic_config is None:
        class_name = modular_config.get("_class_name")
        if isinstance(class_name, str) and hasattr(diffusers_mod, class_name):
            modular_cls = getattr(diffusers_mod, class_name)
        else:
            modular_cls = ModularPipeline

        temp_pipe = ModularPipeline.from_pretrained(
            args.base_model.name, **load_args
        )
        workflow = _resolve_modular_workflow(temp_pipe.blocks, args)
        return ResolvedVideoPipelineSpec(
            kind="modular",
            pipeline_class=type(temp_pipe) if modular_cls is ModularPipeline else modular_cls,
            workflow=workflow,
        )

    if classic_config is None:
        classic_config = DiffusionPipeline.load_config(
            args.base_model.name, **load_args
        )

    class_name = classic_config.get("_class_name")
    if not isinstance(class_name, str):
        raise TaskInvalid("The checkpoint does not declare a pipeline class")

    native_class = getattr(diffusers_mod, class_name, None)
    if native_class is None:
        raise TaskInvalid(f"The checkpoint pipeline class {class_name} is unavailable")

    if issubclass(native_class, ModularPipeline):
        temp_pipe = ModularPipeline.from_pretrained(
            args.base_model.name, **load_args
        )
        workflow = _resolve_modular_workflow(temp_pipe.blocks, args)
        return ResolvedVideoPipelineSpec(
            kind="modular",
            pipeline_class=native_class,
            workflow=workflow,
        )

    if _diffusion_class_supports_mode(native_class, args):
        return ResolvedVideoPipelineSpec(
            kind="diffusion", pipeline_class=native_class
        )

    for candidate_name in _sibling_diffusion_candidates(class_name, input_mode):
        candidate_cls = getattr(diffusers_mod, candidate_name, None)
        if candidate_cls is not None and _diffusion_class_supports_mode(
            candidate_cls, args
        ):
            return ResolvedVideoPipelineSpec(
                kind="diffusion", pipeline_class=candidate_cls
            )

    if input_mode == "i2v":
        mapped_cls = _get_task_class(
            AUTO_IMAGE2VIDEO_PIPELINES_MAPPING,
            class_name,
            throw_error_if_not_exist=False,
        )
        if mapped_cls is not None and _diffusion_class_supports_mode(mapped_cls, args):
            return ResolvedVideoPipelineSpec(
                kind="diffusion", pipeline_class=mapped_cls
            )

    raise TaskInvalid(
        f"Pipeline {class_name} does not support video input mode '{input_mode}'"
    )


def _load_custom_components(
    args: VideoTaskArgs,
    *,
    cache_dir: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
    component_paths: dict[str, str],
) -> dict[str, Any]:
    if not args.components:
        return {}

    pipeline_config = pipeline_config_from_base(
        args.base_model.name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=args.base_model.revision,
    )
    loaded: dict[str, Any] = {}
    for component_name, source in args.components.items():
        format_name = resolve_component_loader(source)
        component_path = component_paths.get(component_name)
        if not component_path:
            raise TaskInvalid(f"Component file for {component_name} is not prepared")
        if format_name == "gguf":
            if component_name not in SUPPORTED_VIDEO_GGUF_COMPONENTS:
                raise TaskInvalid(
                    f"GGUF adapter does not support video component: {component_name}"
                )
            loaded[component_name] = load_gguf_transformer(
                pipeline_config=pipeline_config,
                source=source,
                base_model_name=args.base_model.name,
                base_model_revision=args.base_model.revision,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
                component_path=component_path,
                component_name=component_name,
            )
        elif format_name == DIFFUSERS_COMPONENT_FORMAT:
            loaded[component_name] = load_diffusers_component(
                pipeline_config=pipeline_config,
                component_name=component_name,
                source=source,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
                component_path=component_path,
            )
        else:
            raise TaskInvalid(f"Unsupported component format: {format_name}")
    return loaded


def load_video_pipeline(
    spec: ResolvedVideoPipelineSpec,
    args: VideoTaskArgs,
    *,
    cache_dir: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
    component_paths: dict[str, str] | None = None,
    quantization: dict[str, Any] | None = None,
) -> Any:
    component_paths = component_paths or {}
    custom_components = _load_custom_components(
        args,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        component_paths=component_paths,
    )

    if quantization is not None and quantization["quant_method"] == "bitsandbytes":
        for component_name in quantization["components_to_quantize"]:
            if component_name in custom_components:
                raise TaskInvalid(
                    f"Component {component_name} cannot use both independent "
                    "model files and task quantization"
                )

    if spec.kind == "modular":
        init_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if args.base_model.revision is not None:
            init_kwargs["revision"] = args.base_model.revision
        if spec.workflow is not None:
            init_kwargs["workflow"] = spec.workflow

        pipeline = ModularPipeline.from_pretrained(
            args.base_model.name, **init_kwargs
        )
        if custom_components:
            pipeline.update_components(**custom_components)

        load_comp_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "dtype": torch_dtype,
        }
        if quantization is not None and quantization["quant_method"] == "bitsandbytes":
            from transformers import BitsAndBytesConfig

            pqc = quantization["pipeline_quantization_config"]
            bnb_cfg = BitsAndBytesConfig(**quantization["effective_config"]["quant_kwargs"])
            quant_map: dict[str, Any] = {
                comp_name: bnb_cfg
                for comp_name in quantization["components_to_quantize"]
            }
            load_comp_kwargs["quantization_config"] = quant_map
            _ = pqc

        if spec.workflow is not None:
            pipeline.load_components(workflow=spec.workflow, **load_comp_kwargs)
        else:
            pipeline.load_components(**load_comp_kwargs)
        return pipeline

    load_args: dict[str, Any] = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "dtype": torch_dtype,
    }
    if args.base_model.revision is not None:
        load_args["revision"] = args.base_model.revision
    if quantization is not None and quantization["quant_method"] == "bitsandbytes":
        load_args["quantization_config"] = quantization["pipeline_quantization_config"]

    return spec.pipeline_class.from_pretrained(
        args.base_model.name,
        **load_args,
        **custom_components,
    )


def configure_vae_and_accelerator(pipeline: Any) -> Any:
    vae = getattr(pipeline, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    device = torch.device(utils.get_accelerator())
    return pipeline.to(device)


def load_video_loras(
    pipeline: Any,
    loras: list[VideoLoraArgs],
    *,
    cache_dir: str,
    local_files_only: bool,
) -> None:
    if not loras:
        return
    if not hasattr(pipeline, "load_lora_weights") or not hasattr(
        pipeline, "set_adapters"
    ):
        raise TaskInvalid("The selected video pipeline does not support LoRA")

    load_sig = inspect.signature(pipeline.load_lora_weights).parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in load_sig.values()
    )

    adapter_names: list[str] = []
    adapter_weights: list[float] = []

    def _load_one(
        lora: VideoLoraArgs,
        adapter_name: str,
        *,
        into_transformer_2: bool = False,
        into_transformer_ref: bool = False,
    ) -> None:
        load_kwargs: dict[str, Any] = {
            "adapter_name": adapter_name,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if lora.revision is not None:
            load_kwargs["revision"] = lora.revision
        if lora.weight_file_name:
            load_kwargs["weight_name"] = lora.weight_file_name
        if into_transformer_2:
            if getattr(pipeline, "transformer_2", None) is None:
                raise TaskInvalid("Pipeline does not expose transformer_2 for LoRA target")
            if "load_into_transformer_2" not in load_sig and not accepts_var_kwargs:
                raise TaskInvalid("Pipeline LoRA loader does not support load_into_transformer_2")
            load_kwargs["load_into_transformer_2"] = True
        if into_transformer_ref:
            if getattr(pipeline, "transformer_ref", None) is None:
                raise TaskInvalid("Pipeline does not expose transformer_ref for LoRA target")
            if "load_into_transformer_ref" not in load_sig and not accepts_var_kwargs:
                raise TaskInvalid("Pipeline LoRA loader does not support load_into_transformer_ref")
            load_kwargs["load_into_transformer_ref"] = True
        try:
            pipeline.load_lora_weights(lora.model, **load_kwargs)
        except TaskInvalid:
            raise
        except Exception as exc:
            raise TaskInvalid(
                f"LoRA '{lora.model}' is incompatible with the selected video pipeline"
            ) from exc
        adapter_names.append(adapter_name)
        adapter_weights.append(float(lora.weight))

    for index, lora in enumerate(loras):
        base_adapter = f"lora_{index}"
        target = lora.target
        if target == "auto":
            has_primary = getattr(pipeline, "transformer", None) is not None
            has_ref = getattr(pipeline, "transformer_ref", None) is not None
            _load_one(
                lora,
                base_adapter,
                into_transformer_ref=(not has_primary and has_ref),
            )
        elif target == "transformer":
            if getattr(pipeline, "transformer", None) is None:
                raise TaskInvalid("Pipeline does not expose transformer for LoRA target")
            _load_one(lora, base_adapter)
        elif target == "transformer_2":
            _load_one(lora, base_adapter, into_transformer_2=True)
        elif target == "transformer_ref":
            _load_one(lora, base_adapter, into_transformer_ref=True)
        elif target == "both":
            if getattr(pipeline, "transformer", None) is None or getattr(
                pipeline, "transformer_2", None
            ) is None:
                raise TaskInvalid(
                    "LoRA target 'both' requires both transformer and transformer_2"
                )
            _load_one(lora, f"{base_adapter}_t1", into_transformer_2=False)
            _load_one(lora, f"{base_adapter}_t2", into_transformer_2=True)
        else:
            raise TaskInvalid(f"Unsupported LoRA target: {target}")

    try:
        pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
    except Exception as exc:
        raise TaskInvalid("Failed to activate video LoRA adapters") from exc
