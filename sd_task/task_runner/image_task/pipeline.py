import base64
import inspect
from io import BytesIO
from types import UnionType
from typing import Any, Type, get_args, get_origin, get_type_hints

import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.auto_pipeline import (
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    _get_task_class,
)
from PIL import Image

from sd_task.task_args.image_task import ImageTaskArgs

from .errors import TaskInvalid
from .guidance import (
    add_guidance_call_arg,
    supports_negative_prompt_guidance,
)


def resolve_pipeline_class(
    model_name: str,
    cache_dir: str,
    local_files_only: bool,
    has_images: bool,
    revision: str | None = None,
) -> Type[DiffusionPipeline]:
    load_args: dict[str, Any] = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        load_args["revision"] = revision
    config = DiffusionPipeline.load_config(model_name, **load_args)
    class_name = config.get("_class_name")
    if not isinstance(class_name, str):
        raise TaskInvalid("The checkpoint does not declare a pipeline class")

    native_class = getattr(__import__("diffusers"), class_name, None)
    if native_class is None:
        raise TaskInvalid(f"The checkpoint pipeline class {class_name} is unavailable")
    if not has_images:
        return native_class

    if "image" in inspect.signature(native_class.__call__).parameters:
        return native_class

    image_pipeline_class = _get_task_class(
        AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
        class_name,
        throw_error_if_not_exist=False,
    )
    if image_pipeline_class is not None:
        return image_pipeline_class

    raise TaskInvalid("The selected pipeline does not support image input")


def load_pipeline(
    pipeline_class: Type[DiffusionPipeline],
    args: ImageTaskArgs,
    cache_dir: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
    *,
    component_paths: dict[str, str] | None = None,
    quantization: dict[str, Any] | None = None,
) -> DiffusionPipeline:
    from .quantization.gguf import load_gguf_transformer, pipeline_config_from_base
    from .quantization.registry import resolve_component_loader

    component_paths = component_paths or {}
    load_args: dict[str, Any] = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "dtype": torch_dtype,
    }
    if args.base_model.revision is not None:
        load_args["revision"] = args.base_model.revision

    custom_pipeline_kwargs: dict[str, Any] = {}
    if args.components:
        from .quantization.diffusers_component import (
            DIFFUSERS_COMPONENT_FORMAT,
            load_diffusers_component,
        )

        pipeline_config = pipeline_config_from_base(
            args.base_model.name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=args.base_model.revision,
        )
        for component_name, source in args.components.items():
            format_name = resolve_component_loader(source)
            component_path = component_paths.get(component_name)
            if not component_path:
                raise TaskInvalid(
                    f"Component file for {component_name} is not prepared"
                )
            if format_name == "gguf":
                if component_name != "transformer":
                    raise TaskInvalid(
                        f"GGUF adapter does not support component: {component_name}"
                    )
                custom_pipeline_kwargs[component_name] = load_gguf_transformer(
                    pipeline_config=pipeline_config,
                    source=source,
                    base_model_name=args.base_model.name,
                    base_model_revision=args.base_model.revision,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    local_files_only=local_files_only,
                    component_path=component_path,
                )
            elif format_name == DIFFUSERS_COMPONENT_FORMAT:
                custom_pipeline_kwargs[component_name] = load_diffusers_component(
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

    if quantization is not None and quantization["quant_method"] == "bitsandbytes":
        for component_name in quantization["components_to_quantize"]:
            if component_name in custom_pipeline_kwargs:
                raise TaskInvalid(
                    f"Component {component_name} cannot use both independent "
                    "model files and task quantization"
                )
        load_args["quantization_config"] = quantization[
            "pipeline_quantization_config"
        ]

    return pipeline_class.from_pretrained(
        args.base_model.name,
        **load_args,
        **custom_pipeline_kwargs,
    )


def move_pipeline_to_accelerator(
    pipeline: DiffusionPipeline,
) -> DiffusionPipeline:
    from sd_task import utils

    return pipeline.to(torch.device(utils.get_accelerator()))


def load_lora(
    pipeline: DiffusionPipeline,
    model: str,
    weight_file_name: str,
    weight: float,
    cache_dir: str,
    local_files_only: bool,
) -> None:
    if not hasattr(pipeline, "load_lora_weights") or not hasattr(
        pipeline, "set_adapters"
    ):
        raise TaskInvalid("The selected pipeline does not support LoRA")

    load_args: dict[str, Any] = {
        "adapter_name": "task",
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if weight_file_name:
        load_args["weight_name"] = weight_file_name
    try:
        pipeline.load_lora_weights(model, **load_args)
        pipeline.set_adapters(["task"], adapter_weights=[weight])
    except ValueError as e:
        raise TaskInvalid(
            "The LoRA weights are not compatible with the selected pipeline"
        ) from e


def _annotation_accepts_list(annotation: Any) -> bool:
    if annotation is inspect.Parameter.empty:
        return False
    if isinstance(annotation, str):
        lowered = annotation.lower()
        return "list" in lowered or "sequence" in lowered

    origin = get_origin(annotation)
    if origin in (list, tuple, UnionType):
        return True
    return any(_annotation_accepts_list(arg) for arg in get_args(annotation))


def _decode_image(data_url: str) -> Image.Image:
    header, separator, payload = data_url.partition(",")
    if (
        separator == ""
        or not header.startswith("data:image/")
        or ";base64" not in header
    ):
        raise TaskInvalid("Image inputs must be base64-encoded image data URLs")
    try:
        image_bytes = base64.b64decode(payload, validate=True)
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            return image.copy()
    except Exception as e:
        raise TaskInvalid("Image input is not a valid encoded image") from e


def _pipeline_parameters(pipeline: DiffusionPipeline) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(pipeline.__call__).parameters)


def _image_parameter_accepts_list(
    pipeline: DiffusionPipeline, parameter: inspect.Parameter
) -> bool:
    try:
        annotation = get_type_hints(pipeline.__call__).get(
            "image", parameter.annotation
        )
    except (NameError, TypeError):
        annotation = parameter.annotation
    return _annotation_accepts_list(annotation)


def build_pipeline_call_args(
    pipeline: DiffusionPipeline, args: ImageTaskArgs
) -> dict[str, Any]:
    parameters = _pipeline_parameters(pipeline)
    if "prompt" not in parameters:
        raise TaskInvalid("The selected pipeline does not accept a text prompt")

    candidates: dict[str, Any] = {
        "prompt": args.prompt,
        "width": args.task_config.image_width,
        "height": args.task_config.image_height,
        "num_inference_steps": args.task_config.steps,
        "generator": torch.Generator(device="cpu").manual_seed(args.task_config.seed),
    }
    call_args = {
        name: value for name, value in candidates.items() if name in parameters
    }

    if "num_images_per_prompt" in parameters:
        call_args["num_images_per_prompt"] = args.task_config.num_images_per_prompt
    elif args.task_config.num_images_per_prompt != 1:
        raise TaskInvalid(
            "The selected pipeline does not support multiple output images"
        )

    if args.negative_prompt:
        if not supports_negative_prompt_guidance(
            pipeline,
            parameters,
            args.task_config.guidance_scale,
        ):
            raise TaskInvalid(
                "The selected pipeline does not support effective negative "
                "prompt guidance"
            )
        call_args["negative_prompt"] = args.negative_prompt
    elif args.negative_prompt is not None and "negative_prompt" in parameters:
        call_args["negative_prompt"] = args.negative_prompt

    guidance_parameter = add_guidance_call_arg(
        call_args, pipeline, args.task_config.guidance_scale
    )
    if (
        guidance_parameter == "true_cfg_scale"
        and "negative_prompt" in parameters
        and "negative_prompt" not in call_args
    ):
        call_args["negative_prompt"] = ""

    if args.images:
        image_parameter = parameters.get("image")
        if image_parameter is None:
            raise TaskInvalid("The selected pipeline does not support image input")
        images = [_decode_image(image) for image in args.images]
        if len(images) > 1:
            supports_reference_images = (
                "strength" not in parameters
                and _image_parameter_accepts_list(pipeline, image_parameter)
            )
            if not supports_reference_images:
                raise TaskInvalid(
                    "The selected pipeline does not support multiple "
                    "reference images"
                )
        call_args["image"] = images[0] if len(images) == 1 else images

    if args.strength is not None:
        if "strength" not in parameters:
            raise TaskInvalid("The selected pipeline does not support strength")
        call_args["strength"] = args.strength

    return call_args
