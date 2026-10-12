from typing import Any, Literal

from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated

from ..types import FloatFractionAsInt, NonEmptyString
from ..version import VersionString


class BaseModelArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: NonEmptyString
    revision: NonEmptyString | None = None


class ComponentSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: NonEmptyString
    repo: NonEmptyString
    revision: NonEmptyString | None = None
    format_kwargs: dict[str, Any] = Field(default_factory=dict)


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quant_method: NonEmptyString
    components_to_quantize: list[NonEmptyString]
    quant_kwargs: dict[str, Any]

    @model_validator(mode="after")
    def _validate_components(self) -> "QuantizationConfig":
        if not self.components_to_quantize:
            raise ValueError("components_to_quantize must be non-empty")
        if len(self.components_to_quantize) != len(set(self.components_to_quantize)):
            raise ValueError("components_to_quantize must not contain duplicates")
        return self


class LoraArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: NonEmptyString
    weight_file_name: str = ""
    weight: FloatFractionAsInt = 100


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_width: Annotated[int, Gt(0)] = 1024
    image_height: Annotated[int, Gt(0)] = 1024
    steps: Annotated[int, Gt(0), Le(100)] = 30
    seed: Annotated[int, Ge(0), Lt(2147483648)] = 0
    num_images_per_prompt: Annotated[int, Gt(0), Le(10)] = 1
    guidance_scale: Annotated[float, Ge(0)] | None = None


class ImageTaskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: VersionString = VersionString.V3_7_0
    base_model: BaseModelArgs
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto"
    prompt: NonEmptyString
    negative_prompt: str | None = None
    images: list[NonEmptyString] = Field(default_factory=list)
    strength: Annotated[float, Ge(0), Le(1)] | None = None
    task_config: TaskConfig
    lora: LoraArgs | None = None
    components: dict[str, ComponentSource] = Field(default_factory=dict)
    quantization_config: QuantizationConfig | None = None
