import base64
from io import BytesIO
from pathlib import Path
from shutil import copy2

from PIL import Image

from sd_task.task_args import VideoTaskArgs
from sd_task.task_runner import run_video_task

# Wan 2.1 FLF2V-14B is a single dense 14B DiT (not dual-expert like Wan 2.2 A14B).
# Official checkpoint is 720P; NF4 on transformer + text_encoder targets a single RTX 4090 24GB.
#
# Sample first/last frames are the Diffusers FLF2V bird assets under examples/assets/video.
ASSETS = Path(__file__).resolve().parent / "assets" / "video"
FIRST_FRAME = ASSETS / "flf2v_bird_first_frame.png"
LAST_FRAME = ASSETS / "flf2v_bird_last_frame.png"


def _require_frame(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frame image: {path}")
    return path


def _image_to_data_url(path: Path, size: tuple[int, int]) -> str:
    path = _require_frame(path)
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    rgb.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


# Square sample frames at a 480P-class area to fit a single RTX 4090 24GB.
WIDTH, HEIGHT = 640, 640

args = VideoTaskArgs.model_validate(
    {
        "version": "3.8.0",
        "base_model": {"name": "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers"},
        "dtype": "bfloat16",
        "prompt": (
            "CG animation style, a small blue bird takes off from the ground, "
            "flapping its wings. The bird's feathers are delicate, with a unique "
            "pattern on its chest. The background shows a blue sky with white "
            "clouds under bright sunshine. The camera follows the bird upward, "
            "capturing its flight and the vastness of the sky from a close-up, "
            "low-angle perspective."
        ),
        "negative_prompt": (
            "oversaturated, overexposed, static, blurry, low quality, jpeg artifacts, "
            "deformed, extra limbs, frozen frame, morphing face, sudden cut, "
            "discontinuous motion"
        ),
        "image": _image_to_data_url(FIRST_FRAME, (WIDTH, HEIGHT)),
        "last_image": _image_to_data_url(LAST_FRAME, (WIDTH, HEIGHT)),
        "task_config": {
            "width": WIDTH,
            "height": HEIGHT,
            "num_frames": 81,
            "fps": 16,
            "steps": 30,
            "seed": 0,
            "guidance_scale": 5.5,
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
result = run_video_task(args, output_dir=str(out / "video_wan21_flf2v14b_bnb"))
dest = out / "video_wan21_flf2v14b_bnb.mp4"
copy2(result.output_path, dest)
print("output=", dest)
print("score=", result.score_hex)
print("frames=", result.num_frames, "fps=", result.fps)
print("size=", result.width, "x", result.height)
print("execution_dtype=", result.execution_dtype)
print("execution_quantization=", result.execution_quantization)
