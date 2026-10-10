import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import TaskInvalid


@dataclass(frozen=True)
class GuidanceAdapter:
    matches: Callable[[Any, set[str]], bool]
    parameter: str


def _uses_true_cfg(pipeline: Any, parameters: set[str]) -> bool:
    if not {"guidance_scale", "true_cfg_scale"}.issubset(parameters):
        return False
    transformer = getattr(pipeline, "transformer", None)
    config = getattr(transformer, "config", None)
    return getattr(config, "guidance_embeds", None) is False


def _uses_embedded_guidance(pipeline: Any, parameters: set[str]) -> bool:
    if not {"guidance_scale", "true_cfg_scale"}.issubset(parameters):
        return False
    transformer = getattr(pipeline, "transformer", None)
    config = getattr(transformer, "config", None)
    return getattr(config, "guidance_embeds", None) is True


def _only_true_cfg(_pipeline: Any, parameters: set[str]) -> bool:
    return "true_cfg_scale" in parameters and "guidance_scale" not in parameters


def _only_guidance_scale(_pipeline: Any, parameters: set[str]) -> bool:
    return "guidance_scale" in parameters and "true_cfg_scale" not in parameters


GUIDANCE_ADAPTERS = (
    GuidanceAdapter(_uses_true_cfg, "true_cfg_scale"),
    GuidanceAdapter(_uses_embedded_guidance, "guidance_scale"),
    GuidanceAdapter(_only_true_cfg, "true_cfg_scale"),
    GuidanceAdapter(_only_guidance_scale, "guidance_scale"),
)


def supports_negative_prompt_guidance(
    pipeline: Any,
    parameters: dict[str, inspect.Parameter],
    guidance_scale: float | None,
) -> bool:
    if "negative_prompt" not in parameters:
        return False
    transformer = getattr(pipeline, "transformer", None)
    config = getattr(transformer, "config", None)
    guidance_embeds = getattr(config, "guidance_embeds", None)
    if guidance_embeds is True:
        return False

    parameter_name = "guidance_scale"
    if "true_cfg_scale" in parameters:
        if "guidance_scale" not in parameters or guidance_embeds is False:
            parameter_name = "true_cfg_scale"
    parameter = parameters.get(parameter_name)
    if parameter is None:
        return False

    effective_scale = guidance_scale
    if effective_scale is None:
        effective_scale = parameter.default
    return (
        effective_scale is not inspect.Parameter.empty
        and effective_scale is not None
        and float(effective_scale) > 1
    )


def add_guidance_call_arg(
    call_args: dict[str, Any], pipeline: Any, guidance_scale: float | None
) -> str | None:
    if guidance_scale is None:
        return None

    parameters = set(inspect.signature(pipeline.__call__).parameters)
    for adapter in GUIDANCE_ADAPTERS:
        if adapter.matches(pipeline, parameters):
            call_args[adapter.parameter] = guidance_scale
            return adapter.parameter

    raise TaskInvalid(
        "The selected pipeline does not expose an unambiguous guidance parameter"
    )
