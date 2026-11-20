import base64
from io import BytesIO
from pathlib import Path
from shutil import copy2

from PIL import Image, ImageDraw

from sd_task.task_args import VideoTaskArgs
from sd_task.task_runner import run_video_task


def _sample_start_frame_data_url() -> str:
    image = Image.new("RGB", (832, 480), color=(72, 128, 180))
    draw = ImageDraw.Draw(image)
    draw.ellipse((280, 140, 552, 340), fill=(240, 210, 90))
    draw.rectangle((360, 300, 472, 420), fill=(60, 140, 70))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


# Same TI2V-5B weights; I2V is selected by supplying `image`.
args = VideoTaskArgs.model_validate(
    {
        "version": "3.8.0",
        "base_model": {"name": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"},
        "dtype": "bfloat16",
        "prompt": (
            "The yellow sun slowly rises over a calm blue lake, gentle ripples on "
            "the water, soft morning mist, smooth camera push-in"
        ),
        "negative_prompt": (
            "oversaturated, overexposed, static, blurry, low quality, jpeg artifacts, "
            "deformed, frozen frame, cluttered background"
        ),
        "image": _sample_start_frame_data_url(),
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
result = run_video_task(args, output_dir=str(out / "video_wan22_ti2v5b_i2v_bnb"))
dest = out / "video_wan22_ti2v5b_i2v_bnb.mp4"
copy2(result.output_path, dest)
print("output=", dest)
print("frames=", result.num_frames, "fps=", result.fps)
print("size=", result.width, "x", result.height)
print("execution_dtype=", result.execution_dtype)
print("execution_quantization=", result.execution_quantization)
