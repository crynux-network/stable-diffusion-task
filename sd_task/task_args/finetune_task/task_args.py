from typing import Annotated, Literal, List

from annotated_types import Ge, Gt, MinLen, Le
from pydantic import BaseModel, HttpUrl

from ..types import NonEmptyString


class TransformArgs(BaseModel):
    center_crop: bool = False
    random_flip: bool = False


class LRSchedulerArgs(BaseModel):
    lr_scheduler: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = "constant"
    lr_warmup_steps: int = 500


class AdamOptimizerArgs(BaseModel):
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-2
    epsilon: float = 1e-8


class LoraArgs(BaseModel):
    rank: int = 8
    init_lora_weights: bool | Literal["gaussian", "loftq"] = True
    target_modules: List[str] | str | None = None


class ModelArgs(BaseModel):
    name: Annotated[str, MinLen(1)]
    variant: str | None = None
    revision: str = "main"


class DatasetArgs(BaseModel):
    url: HttpUrl | None = None
    name: Annotated[str, MinLen(1)] | None = None
    config_name: str | None = None
    image_column: str = "image"
    caption_column: str = "text"


class ValidationArgs(BaseModel):
    prompt: str | None = None
    num_images: Annotated[int, Ge(1), Le(10)] = 4


class TrainArgs(BaseModel):
    learning_rate: Annotated[float, Gt(0)] = 1e-4
    batch_size: Annotated[int, Ge(1)] = 16
    gradient_accumulation_steps: Annotated[int, Ge(1)] = 1
    prediction_type: Literal["epsilon", "v_prediction"] | None = None
    max_grad_norm: float = 1.0
    num_train_epochs: Annotated[int, Ge(1)] = 1
    num_train_steps: Annotated[int, Ge(1)] | None = None
    max_train_epochs: Annotated[int, Ge(1)] = 1
    max_train_steps: Annotated[int, Ge(1)] | None = None
    scale_lr: bool = True
    resolution: Annotated[int, Ge(1)] = 512
    noise_offset: float = 0
    snr_gamma: float | None = None

    lr_scheduler: LRSchedulerArgs = LRSchedulerArgs()
    adam_args: AdamOptimizerArgs = AdamOptimizerArgs()


class FinetuneLoraTaskArgs(BaseModel):
    model: ModelArgs
    dataset: DatasetArgs
    validation: ValidationArgs
    train_args: TrainArgs
    lora: LoraArgs = LoraArgs()

    transforms: TransformArgs = TransformArgs()
    dataloader_num_workers: int = 0

    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    seed: int = 0

    checkpoint: Annotated[str, MinLen(1)] | None = None

    version: NonEmptyString = "2.5.0"
