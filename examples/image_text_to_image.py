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

run_image_task(args)[0].save("./data/image_text_to_image.png")
