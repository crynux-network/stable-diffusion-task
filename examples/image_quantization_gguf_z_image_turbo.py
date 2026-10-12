from pathlib import Path

from sd_task.task_args import ImageTaskArgs
from sd_task.task_runner import run_image_task

args = ImageTaskArgs.model_validate(
    {
        "version": "3.7.0",
        "base_model": {"name": "Tongyi-MAI/Z-Image-Turbo"},
        "dtype": "bfloat16",
        "prompt": "A product photo of a white ceramic cup on a wooden table",
        "task_config": {
            "image_width": 1024,
            "image_height": 1024,
            "steps": 9,
            "seed": 0,
            "num_images_per_prompt": 1,
        },
        "components": {
            "transformer": {
                "format": "gguf",
                "repo": "jayn7/Z-Image-Turbo-GGUF",
                "revision": None,
                "format_kwargs": {
                    "filename": "z_image_turbo-Q4_K_M.gguf",
                },
            }
        },
    }
)

out = Path("./data")
out.mkdir(parents=True, exist_ok=True)
result = run_image_task(args)
result[0].save(out / "image_quantization_gguf_z_image_turbo.png")
print("execution_dtype=", result.execution_dtype)
print("execution_quantization=", result.execution_quantization)
