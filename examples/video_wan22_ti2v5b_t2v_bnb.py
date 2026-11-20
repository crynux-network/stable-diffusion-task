from pathlib import Path
from shutil import copy2

from sd_task.task_args import VideoTaskArgs
from sd_task.task_runner import run_video_task

# Wan 2.2 TI2V-5B is the Wan 2.2 variant that fits a single RTX 4090 24GB
# without CPU offload when the transformer and T5 text encoder are NF4-quantized.
args = VideoTaskArgs.model_validate(
    {
        "version": "3.8.0",
        "base_model": {"name": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"},
        "dtype": "bfloat16",
        "prompt": (
            "A golden retriever runs across a sunny meadow, camera tracking from "
            "the side, soft wind in the grass, cinematic natural lighting"
        ),
        "negative_prompt": (
            "oversaturated, overexposed, static, blurry, low quality, jpeg artifacts, "
            "deformed, extra limbs, frozen frame, cluttered background"
        ),
        "task_config": {
            "width": 832,
            "height": 480,
            "num_frames": 49,
            "fps": 24,
            "steps": 20,
            "seed": 0,
            "guidance_scale": 5.0,
        },
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
result = run_video_task(args, output_dir=str(out / "video_wan22_ti2v5b_t2v_bnb"))
dest = out / "video_wan22_ti2v5b_t2v_bnb.mp4"
copy2(result.output_path, dest)
print("output=", dest)
print("frames=", result.num_frames, "fps=", result.fps)
print("size=", result.width, "x", result.height)
print("execution_dtype=", result.execution_dtype)
print("execution_quantization=", result.execution_quantization)
