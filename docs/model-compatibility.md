# Image Model Compatibility

## Authority and scope

This document specifies image execution from validated `ImageTaskArgs` to the
ordered list of generated PIL images.

Quantized model loading, task-time quantization, platform restrictions,
LoRA compatibility, cache identity, and verification requirements are
specified in [quantization.md](quantization.md).

Public HTTP request conversion and public response conversion MUST remain
outside this package. The package MUST NOT implement OpenAI Images or Responses
API semantics.

## Common contract

Every checkpoint MUST use the same `ImageTaskArgs` input and
`ImageTaskResult` output.

The output MUST contain every image returned by the pipeline in pipeline order.
The number of output images MUST equal
`task_config.num_images_per_prompt`.

The core execution path MUST NOT select behavior from a model ID, repository
path, model name, or hard-coded pipeline class name. Model names in this
document are verification samples and MUST NOT become a runtime allowlist.

Unknown task fields MUST fail validation.

## Processing stages

### 1. Task validation

`ImageTaskArgs` MUST validate:

- one base pipeline repository;
- one runtime dtype;
- one prompt and an optional negative prompt;
- zero or more base64 image data URLs;
- optional image-to-image strength;
- dimensions, steps, seed, output count, and optional guidance scale; and
- one optional LoRA adapter.

An empty `images` list SHALL select text-to-image execution. A non-empty list
SHALL select image editing or image-to-image execution.

### 2. Pipeline resolution

The runtime MUST read `model_index.json` before loading any model component or
allocating model weights in VRAM.

For text-to-image execution, the runtime MUST use the checkpoint's declared
pipeline class.

For image input, the runtime MUST first inspect the checkpoint's declared
pipeline class. If its call signature accepts `image`, the runtime MUST use
that native class. This path covers unified pipelines that implement
text-to-image and image editing through the same class.

Only when the declared pipeline does not accept `image` SHALL the runtime
resolve a separate image-to-image class through the Diffusers
`AUTO_IMAGE2IMAGE_PIPELINES_MAPPING`. This fallback covers checkpoints that
still separate text-to-image and image-to-image pipelines.

Resolution failure MUST be reported before `from_pretrained` is called. The
runtime MUST NOT load a text-to-image pipeline and then convert or reload it as
an image-to-image pipeline.

### 3. Artifact loading

The resolved pipeline class MUST call `from_pretrained` exactly once on a cache
miss. The checkpoint scheduler MUST be retained.

`dtype` MUST control the loaded PyTorch runtime precision. It MUST NOT select a
repository filename variant, FP8 artifact, or quantized repository.

Before any weights are loaded, the runtime MUST resolve TaskArgs `dtype` to one
concrete value among `float16`, `bfloat16`, and `float32`. Explicit concrete
values MUST be kept unchanged. When `dtype` is `auto`, the runtime MUST read the
base pipeline `transformer/config.json` field `torch_dtype`. A missing, `null`,
or `auto` config value MUST resolve to `float32`. Any other non-empty illegal
config value MUST fail as an invalid task. Resolution MUST NOT branch on model
ID or node GPU model. The same concrete dtype MUST be passed to the base
pipeline load, GGUF `compute_dtype` / `from_single_file`, independent Diffusers
components, and the default BitsAndBytes `bnb_4bit_compute_dtype` when the
client omits that field. The model cache key MUST use the resolved concrete
dtype, not the literal `auto`.

When LoRA is present, the pipeline MUST expose the standard Diffusers
`load_lora_weights` and `set_adapters` interfaces. An incompatible LoRA MUST be
reported as an invalid task.

### 4. Input adaptation

The core call arguments MUST be derived from the resolved pipeline signature.
Unsupported optional arguments MUST NOT be forwarded.

The runtime MUST pass `num_images_per_prompt` to pipelines that support it. A
pipeline without this parameter MUST reject values greater than one.

A single input image MUST be passed as one PIL image. Multiple input images
MUST be passed as a PIL list only when the pipeline accepts a list and does not
expose `strength`. A pipeline that exposes `strength` treats an image list as a
batch rather than multiple references and MUST reject multiple task images.

`strength` MUST be forwarded only when the resolved pipeline exposes it.

### 5. Guidance adaptation

`task_config.guidance_scale` is the only public guidance field.

When the pipeline exposes only `guidance_scale`, the value MUST map to
`guidance_scale`. When it exposes only `true_cfg_scale`, the value MUST map to
`true_cfg_scale`.

When both parameters exist, `transformer.config.guidance_embeds=False` MUST
select `true_cfg_scale`, and `guidance_embeds=True` MUST select
`guidance_scale`. A missing or ambiguous capability value MUST fail before
generation. Both parameters MUST NOT be sent together.

A selected `true_cfg_scale` MUST include an empty negative prompt when the task
does not provide one, because true CFG requires the unconditional prompt path.

A non-empty `negative_prompt` MUST be accepted only when the pipeline exposes
`negative_prompt` and the loaded transformer does not declare embedded
guidance. Its effective true CFG or CFG scale MUST be greater than one.
Unsupported or ineffective negative prompts MUST fail before generation.

### 6. Generation and output

Generation MUST use one pipeline call. The runtime MUST return all
`result.images`.

The reported execution dtype MUST come from an actual
`pipeline.transformer` parameter. A pipeline without `transformer` MUST fail.

## Cache identity

The model cache key MUST contain:

- the resolved pipeline class;
- the base model;
- the runtime dtype; and
- the complete LoRA configuration.

Text-to-image and image-input tasks MUST share a cache entry when they resolve
to the same class. They MUST use different entries when they resolve to
different classes.

## Verified coverage samples

The evidence boundary for each row is the linked official model card or
Diffusers pipeline documentation. A feature marked `Not verified` MUST NOT be
inferred from another model in the same family.

Recommended VRAM assumes the default BF16 or FP16 checkpoint without
quantization or CPU offload.

| Sample | Recommended VRAM | Resolved execution | Image input | Guidance | Negative prompt | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Stable Diffusion 3.5 | [24 GB](https://willitrunai.com/image-models/sd-3-5-large) | `StableDiffusion3Pipeline`; AutoPipeline resolves `StableDiffusion3Img2ImgPipeline` for image input | One img2img image with `strength`; multiple reference images are not supported | `guidance_scale` | Supported by CFG | [Diffusers SD3](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3) |
| Z-Image Base | [16 GB](https://vramready.com/models/z-image/) | `ZImagePipeline`; AutoPipeline resolves `ZImageImg2ImgPipeline` for image input | One img2img image with `strength`; multiple reference images are not supported | `guidance_scale` | Supported by CFG | [Diffusers Z-Image](https://huggingface.co/docs/diffusers/api/pipelines/z_image) |
| Z-Image Turbo | [16 GB](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | `ZImagePipeline`; AutoPipeline resolves `ZImageImg2ImgPipeline` for image input | One img2img image with `strength`; multiple reference images are not supported | Pipeline default or `guidance_scale` when effective | Not supported by the distilled checkpoint | [Z-Image Turbo model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |
| FLUX.2 klein 4B | [16 GB](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | The native `Flux2KleinPipeline` handles text and image input | One or more reference images; no `strength` | `guidance_scale` | Not supported | [Diffusers FLUX.2](https://huggingface.co/docs/diffusers/api/pipelines/flux2) |
| FLUX.2 klein 9B | [32 GB](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | The native `Flux2KleinPipeline` handles text and image input | One or more reference images; no `strength` | `guidance_scale` | Not supported | [Diffusers FLUX.2](https://huggingface.co/docs/diffusers/api/pipelines/flux2) |
| FLUX.2 dev | [96 GB](https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_dev_hf.md) | The native `Flux2Pipeline` handles text and image input | One or more reference images; no `strength` | `guidance_scale` | Not supported | [FLUX.2 dev model card](https://huggingface.co/black-forest-labs/FLUX.2-dev) |
| Qwen-Image and Qwen-Image-2512 | [48 GB](https://www.spheron.network/tools/gpu-recommender/Qwen/Qwen-Image-2512/) | Native text-to-image pipeline; AutoPipeline resolves the registered Qwen image-to-image pipeline for image input | One img2img image; multiple-reference editing is not verified for the base checkpoint | `true_cfg_scale` when `guidance_embeds=False` | Supported with true CFG | [Diffusers QwenImage](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage) |
| Qwen-Image-Edit | [48 GB](https://www.thundercompute.com/blog/qwen-image-edit-comfyui) | The checkpoint's native edit pipeline | One or more edit images when its signature accepts a list; no `strength` | `true_cfg_scale` when `guidance_embeds=False` | Supported with true CFG | [Qwen-Image-Edit model card](https://huggingface.co/Qwen/Qwen-Image-Edit) |
| Qwen-Image-2.1 | [48 GB](https://www.genvidkit.com/blog/qwen-image-2-1-vram-requirements) | The native unified pipeline handles text and image input | One or more reference images; no `strength` | Capability-selected `true_cfg_scale` or `guidance_scale` | Supported only when true CFG is selected | [Qwen-Image-2.1 model card](https://huggingface.co/Qwen/Qwen-Image-2.1) |

LoRA loading is verified through the standard Diffusers loader contract.
Per-checkpoint LoRA weight compatibility remains an artifact validation step
and MUST NOT be inferred from base pipeline compatibility.
