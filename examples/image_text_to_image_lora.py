from sd_task.task_args import ImageTaskArgs
from sd_task.task_runner import run_image_task


args = ImageTaskArgs.model_validate(
    {
        "version": "3.7.0",
        "base_model": {"name": "Qwen/Qwen-Image"},
        "dtype": "bfloat16",
        "prompt": "Valentin standing at a cafe entrance in natural daylight",
        "negative_prompt": "blurry, low quality",
        "task_config": {
            "image_width": 1024,
            "image_height": 1024,
            "steps": 50,
            "seed": 0,
            "num_images_per_prompt": 1,
            "guidance_scale": 4,
        },
        "lora": {
            "model": "flymy-ai/qwen-image-lora",
            "weight_file_name": "pytorch_lora_weights.safetensors",
            "weight": 100,
        },
    }
)

run_image_task(args)[0].save("./data/image_text_to_image_lora.png")
