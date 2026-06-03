import re

from compel import Compel, ReturnedEmbeddingsType
from compel.convenience_wrappers import CompelForSDXL


def add_prompt_pipeline_call_args(call_args, pipeline, prompt: str, negative_prompt: str):
    class_name = type(pipeline).__name__

    if re.match(r"StableDiffusionXL", class_name) is None:
        add_prompt_pipeline_sd15_call_args(call_args, pipeline, prompt, negative_prompt)
    else:
        add_prompt_pipeline_sdxl_call_args(call_args, pipeline, prompt, negative_prompt)


def add_prompt_pipeline_sdxl_call_args(call_args, pipeline, prompt: str, negative_prompt: str):
    conditioning = CompelForSDXL(pipeline)(prompt, negative_prompt=negative_prompt)

    call_args["prompt_embeds"] = conditioning.embeds
    call_args["pooled_prompt_embeds"] = conditioning.pooled_embeds
    call_args["negative_prompt_embeds"] = conditioning.negative_embeds
    call_args["negative_pooled_prompt_embeds"] = conditioning.negative_pooled_embeds


def add_prompt_refiner_sdxl_call_args(call_args, refiner, prompt: str, negative_prompt: str):
    compel = Compel(
        tokenizer=refiner.tokenizer_2,
        text_encoder=refiner.text_encoder_2,
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=True,
        truncate_long_prompts=False,
    )

    conditioning, pooled = compel([prompt, negative_prompt])

    call_args["prompt_embeds"] = conditioning[0:1]
    call_args["pooled_prompt_embeds"] = pooled[0:1]
    call_args["negative_prompt_embeds"] = conditioning[1:2]
    call_args["negative_pooled_prompt_embeds"] = pooled[1:2]


def add_prompt_pipeline_sd15_call_args(call_args, pipeline, prompt: str, negative_prompt: str):
    compel = Compel(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        requires_pooled=False,
        truncate_long_prompts=False,
    )

    conditioning = compel([prompt, negative_prompt])

    call_args["prompt_embeds"] = conditioning[0:1]
    call_args["negative_prompt_embeds"] = conditioning[1:2]
