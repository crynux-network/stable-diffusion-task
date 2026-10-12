from pathlib import Path

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
            "steps": 8,
            "seed": 0,
            "num_images_per_prompt": 1,
        },
        "components": {},
        "quantization_config": {
            "quant_method": "bitsandbytes",
            "components_to_quantize": ["transformer", "text_encoder"],
            "quant_kwargs": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": False,
            },
        },
    }
)

out = Path("./data")
out.mkdir(parents=True, exist_ok=True)
result = run_image_task(args)
result[0].save(out / "image_quantization_bnb_qwen_image_21.png")
print("execution_dtype=", result.execution_dtype)
print("execution_quantization=", result.execution_quantization)
