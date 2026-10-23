# Image Task

This package defines and executes Crynux image generation and image editing
tasks through Diffusers pipelines.

## Execution contract

All supported checkpoints use `ImageTaskArgs` and return an `ImageTaskResult`.
The result remains list-compatible and contains every generated image in
pipeline order.

The runtime:

- resolves the final pipeline class from `model_index.json` before loading
  model weights;
- uses the declared pipeline directly when it accepts image input and resolves
  a separate AutoPipeline image-to-image class only as a fallback;
- loads one pipeline for each cache entry;
- uses the checkpoint scheduler;
- forwards only parameters accepted by the resolved pipeline;
- supports text-to-image and capability-detected image input;
- maps the common `guidance_scale` field to the pipeline's effective guidance
  parameter;
- supports compatible LoRA adapters; and
- does not select behavior from model IDs or repository paths.

The complete compatibility contract is in
[`docs/model-compatibility.md`](docs/model-compatibility.md).

Fine-tune LoRA, tagger, background removal, and ControlNet preprocessing remain
separate task runners with their existing contracts.

## Example

```python
from sd_task.task_args import ImageTaskArgs
from sd_task.task_runner import run_image_task

args = ImageTaskArgs.model_validate(
    {
        "version": "3.7.0",
        "base_model": {"name": "Qwen/Qwen-Image-2.1"},
        "dtype": "bfloat16",
        "prompt": "A product photo of a white ceramic cup on a wooden table",
        "task_config": {
            "image_width": 1024,
            "image_height": 1024,
            "steps": 30,
            "seed": 0,
            "num_images_per_prompt": 1,
            "guidance_scale": 4,
        },
    }
)

images = run_image_task(args)
images[0].save("./data/result.png")
```

Text-to-image, LoRA, and image-editing examples are available under
[`examples`](examples).

## Configuration

`preloaded_models.base` lists pipeline repositories that `sd_task/prefetch.py`
downloads into the configured Hugging Face cache. Runtime `dtype` controls the
loaded PyTorch precision. `auto` is resolved to a concrete dtype from the base
pipeline transformer config before weights load. It does not select repository
weight variants or quantized artifacts.

## JSON schema

Task schemas are stored under [`schema`](schema). Regenerate them with:

```shell
python ./generate_json_schema.py
```
