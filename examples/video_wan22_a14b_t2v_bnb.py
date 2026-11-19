from pathlib import Path
from shutil import copy2

from sd_task.task_args import VideoTaskArgs
from sd_task.task_runner import run_video_task

# Wan 2.2 A14B keeps two 14B experts resident (transformer + transformer_2).
# NF4 on both denoisers targets ~22-26 GB peak and may still OOM on a single
# 24GB card without CPU offload; prefer TI2V-5B for single-GPU 4090 smoke tests.
args = VideoTaskArgs.model_validate(
    {
        "version": "3.8.0",
        "base_model": {"name": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"},
        "dtype": "bfloat16",
        "prompt": (
            "Two anthropomorphic cats in comfy boxing gear fight on a spotlighted "
            "stage, dynamic camera orbit, cinematic lighting"
        ),
        "negative_prompt": (
            "oversaturated, overexposed, static, blurry, low quality, jpeg artifacts, "
            "deformed, extra limbs, frozen frame, cluttered background"
        ),
        "task_config": {
            "width": 832,
            "height": 480,
            "num_frames": 81,
            "fps": 16,
            "steps": 40,
            "seed": 0,
            "guidance_scale": 4.0,
        },
        "pipeline_args": {
            "guidance_scale_2": 3.0,
        },
        "quantization_config": {
            "quant_method": "bitsandbytes",
            "components_to_quantize": ["transformer", "transformer_2", "text_encoder"],
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
result = run_video_task(args, output_dir=str(out / "video_wan22_a14b_t2v_bnb"))
dest = out / "video_wan22_a14b_t2v_bnb.mp4"
copy2(result.output_path, dest)
print("output=", dest)
print("score=", result.score_hex)
print("frames=", result.num_frames, "fps=", result.fps)
print("size=", result.width, "x", result.height)
print("execution_dtype=", result.execution_dtype)
print("execution_quantization=", result.execution_quantization)
