from sd_task.task_args import ImageTaskArgs
from sd_task.task_runner import run_image_task


args = ImageTaskArgs.model_validate(
    {
        "version": "3.7.0",
        "base_model": {"name": "black-forest-labs/FLUX.2-klein-4B"},
        "dtype": "bfloat16",
        "prompt": "A product photo of a white ceramic cup on a wooden table",
        "task_config": {
            "image_width": 1024,
            "image_height": 1024,
            "steps": 4,
            "seed": 0,
            "num_images_per_prompt": 1,
            "guidance_scale": 1,
        },
    }
)

run_image_task(args)[0].save("./data/image_flux2_klein_4b_text_to_image.png")
