from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import av
import numpy as np
import pytest
import torch
from PIL import Image

from sd_task.cache import MemoryModelCache
from sd_task.config import Config
from sd_task.task_args.video_task import VideoLoraArgs, VideoTaskArgs
from sd_task.task_runner.image_task.errors import (
    ExecutionModelFileError,
    TaskInvalid,
)
from sd_task.task_runner.video_task import (
    apply_execution_model_files,
    run_video_task,
)
from sd_task.task_runner.video_task.encode import encode_canonical_mp4
from sd_task.task_runner.video_task.input_adapter import (
    build_video_call_args,
    validate_geometry_and_temporal_grid,
)
from sd_task.task_runner.video_task.pipeline import (
    ResolvedVideoPipelineSpec,
    _resolve_modular_workflow,
    load_video_loras,
)
from sd_task.utils import encode_image_dataurl


def _png_data_url(width: int = 64, height: int = 64, color: tuple[int, int, int] = (120, 40, 200)) -> str:
    img = Image.new("RGB", (width, height), color=color)
    return encode_image_dataurl(img)


def test_encode_canonical_mp4_deterministic_silent_and_audio(tmp_path):
    num_frames, height, width = 17, 64, 96
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    for idx in range(num_frames):
        yy = np.arange(height, dtype=np.uint8)[:, None]
        xx = np.arange(width, dtype=np.uint8)[None, :]
        frames[idx, :, :, 0] = (xx * 2 + idx * 9) % 256
        frames[idx, :, :, 1] = (yy * 3 + idx * 5) % 256
        frames[idx, :, :, 2] = ((xx + yy) + idx * 11) % 256

    silent_mp4 = str(tmp_path / "silent_0.mp4")
    silent_mp4_repeat = str(tmp_path / "silent_1.mp4")
    n1, w1, h1, a1 = encode_canonical_mp4(frames, fps=24.0, output_path=silent_mp4)
    n2, w2, h2, a2 = encode_canonical_mp4(frames, fps=24.0, output_path=silent_mp4_repeat)

    assert (n1, w1, h1, a1) == (17, 96, 64, False)
    assert (n2, w2, h2, a2) == (17, 96, 64, False)

    with av.open(silent_mp4) as c1, av.open(silent_mp4_repeat) as c2:
        assert len(c1.streams.audio) == 0
        v1 = [bytes(f.planes[0]) for f in c1.decode(c1.streams.video[0])]
        v2 = [bytes(f.planes[0]) for f in c2.decode(c2.streams.video[0])]
        assert len(v1) == 17
        assert v1 == v2

    sample_rate = 24000
    t = np.arange(sample_rate, dtype=np.float32) / float(sample_rate)
    stereo = np.stack(
        [
            0.5 * np.sin(2.0 * np.pi * 440.0 * t),
            0.4 * np.cos(2.0 * np.pi * 880.0 * t),
        ],
        axis=0,
    )
    av_mp4 = str(tmp_path / "av_0.mp4")
    n3, w3, h3, a3 = encode_canonical_mp4(
        frames,
        fps=24.0,
        output_path=av_mp4,
        audio=torch.from_numpy(stereo),
        audio_sample_rate=sample_rate,
    )
    assert (n3, w3, h3, a3) == (17, 96, 64, True)
    with av.open(av_mp4) as c3:
        assert len(c3.streams.video) == 1
        assert len(c3.streams.audio) == 1
        assert len(c3.streams.audio[0].layout.channels) == 2


def test_apply_execution_model_files_pins_base_and_components():
    commit_base = "a" * 40
    commit_te = "b" * 40
    args = VideoTaskArgs.model_validate(
        {
            "version": "3.8.0",
            "base_model": {"name": "wan-ai/wan2.2-ti2v-5b-diffusers"},
            "prompt": "Ocean waves under moonlight",
            "components": {
                "text_encoder": {
                    "format": "diffusers",
                    "repo": "org/custom-umt5",
                    "format_kwargs": {"subfolder": "text_encoder"},
                }
            },
            "loras": [
                {
                    "model": "lightx2v/wan2.2-lightning",
                    "weight_file_name": "fast.safetensors",
                }
            ],
        }
    )
    entries = [
        SimpleNamespace(
            repository="wan-ai/wan2.2-ti2v-5b-diffusers",
            revision=commit_base,
            format="diffusers",
        ),
        SimpleNamespace(
            repository="org/custom-umt5",
            revision=commit_te,
            format="diffusers",
        ),
    ]
    pinned = apply_execution_model_files(args, entries)
    assert pinned.base_model.revision == commit_base
    assert pinned.components["text_encoder"].revision == commit_te

    conflicting = args.model_copy(deep=True)
    conflicting.base_model.revision = "c" * 40
    with pytest.raises(ExecutionModelFileError):
        apply_execution_model_files(conflicting, entries)


def test_spatial_and_temporal_grid_validation_rejects_misaligned_dims():
    fake_wan = SimpleNamespace(
        vae_scale_factor_spatial=8,
        vae_scale_factor_temporal=4,
        transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
    )
    bad_width_args = VideoTaskArgs.model_validate(
        {
            "version": "3.8.0",
            "base_model": {"name": "wan-ai/wan2.2"},
            "prompt": "Test",
            "task_config": {"width": 710, "height": 1280, "num_frames": 81},
        }
    )
    with pytest.raises(TaskInvalid, match="divisible by 16"):
        validate_geometry_and_temporal_grid(fake_wan, bad_width_args)

    bad_frames_args = VideoTaskArgs.model_validate(
        {
            "version": "3.8.0",
            "base_model": {"name": "wan-ai/wan2.2"},
            "prompt": "Test",
            "task_config": {"width": 720, "height": 1280, "num_frames": 80},
        }
    )
    with pytest.raises(TaskInvalid, match="% 4 == 0"):
        validate_geometry_and_temporal_grid(fake_wan, bad_frames_args)

    fake_h3 = SimpleNamespace(
        canvas_multiple=32,
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        fps=24,
        min_duration=5.0,
        max_duration=15.0,
    )
    bad_h3_frames = VideoTaskArgs.model_validate(
        {
            "version": "3.8.0",
            "base_model": {"name": "minimaxai/minimax-h3"},
            "prompt": "Test",
            "task_config": {"width": 768, "height": 1344, "num_frames": 121},
        }
    )
    with pytest.raises(TaskInvalid, match="% 17 == 5"):
        validate_geometry_and_temporal_grid(fake_h3, bad_h3_frames)

    good_h3_frames = VideoTaskArgs.model_validate(
        {
            "version": "3.8.0",
            "base_model": {"name": "minimaxai/minimax-h3"},
            "prompt": "Test",
            "task_config": {"width": 768, "height": 1344, "num_frames": 124},
        }
    )
    validate_geometry_and_temporal_grid(fake_h3, good_h3_frames)


def test_modular_workflow_resolution_for_h3_and_ltx2():
    h3_blocks = SimpleNamespace(
        inputs=[
            SimpleNamespace(name=n)
            for n in ("prompt", "image", "last_image", "references", "num_frames")
        ],
        _workflow_map={
            "t2va": {"prompt": True},
            "fl2va": (
                {"prompt": True, "image": True},
                {"prompt": True, "last_image": True},
            ),
            "ref2va": {"prompt": True, "references": True},
        },
    )
    img_url = _png_data_url()
    assert (
        _resolve_modular_workflow(
            h3_blocks,
            VideoTaskArgs.model_validate(
                {"version": "3.8.0", "base_model": {"name": "m/h3"}, "prompt": "hi"}
            ),
        )
        == "t2va"
    )
    assert (
        _resolve_modular_workflow(
            h3_blocks,
            VideoTaskArgs.model_validate(
                {
                    "version": "3.8.0",
                    "base_model": {"name": "m/h3"},
                    "prompt": "hi",
                    "image": img_url,
                    "last_image": img_url,
                }
            ),
        )
        == "fl2va"
    )
    assert (
        _resolve_modular_workflow(
            h3_blocks,
            VideoTaskArgs.model_validate(
                {
                    "version": "3.8.0",
                    "base_model": {"name": "m/h3"},
                    "prompt": "hi",
                    "references": [{"type": "image", "data": img_url}],
                }
            ),
        )
        == "ref2va"
    )

    ltx_blocks = SimpleNamespace(
        inputs=[
            SimpleNamespace(name=n)
            for n in ("prompt", "image", "conditions", "reference_conditions", "num_frames")
        ],
        _workflow_map={
            "text2video": {"prompt": True},
            "image2video": {"image": True, "prompt": True},
            "condition": {"conditions": True, "prompt": True},
            "in_context": {
                "reference_conditions": True,
                "num_frames": True,
                "prompt": True,
            },
        },
    )
    assert (
        _resolve_modular_workflow(
            ltx_blocks,
            VideoTaskArgs.model_validate(
                {
                    "version": "3.8.0",
                    "base_model": {"name": "l/ltx2"},
                    "prompt": "hi",
                    "image": img_url,
                    "last_image": img_url,
                }
            ),
        )
        == "condition"
    )


def test_multi_lora_dual_denoiser_routing():
    calls: list[tuple[str, dict[str, Any]]] = []
    activated: list[tuple[list[str], list[float]]] = []

    class DummyDualPipeline:
        transformer = object()
        transformer_2 = object()

        def load_lora_weights(self, model: str, **kwargs: Any) -> None:
            calls.append((model, kwargs))

        def set_adapters(
            self, adapter_names: list[str], adapter_weights: list[float]
        ) -> None:
            activated.append((adapter_names, adapter_weights))

    pipe = DummyDualPipeline()
    loras = [
        VideoLoraArgs(
            model="org/speed",
            weight_file_name="high.safetensors",
            weight=100,
            target="transformer",
        ),
        VideoLoraArgs(
            model="org/speed",
            weight_file_name="low.safetensors",
            weight=100,
            target="transformer_2",
        ),
        VideoLoraArgs(
            model="org/char",
            weight_file_name="char.safetensors",
            weight=80,
            target="both",
        ),
    ]
    load_video_loras(pipe, loras, cache_dir="/tmp", local_files_only=True)
    assert len(calls) == 4
    assert calls[0][1].get("load_into_transformer_2", False) is False
    assert calls[1][1].get("load_into_transformer_2") is True
    assert calls[2][1]["adapter_name"] == "lora_2_t1"
    assert calls[3][1]["adapter_name"] == "lora_2_t2"
    assert calls[3][1]["load_into_transformer_2"] is True
    assert activated == [
        (["lora_0", "lora_1", "lora_2_t1", "lora_2_t2"], [1.0, 1.0, 0.8, 0.8])
    ]


def test_run_video_task_end_to_end_with_stub_pipeline(tmp_path, monkeypatch):
    @dataclass
    class DummyOutput:
        frames: np.ndarray

    class DummyDenoiser(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.config = SimpleNamespace(patch_size=(1, 2, 2), guidance_embeds=False)

    class DummyVAE:
        spatial_compression_ratio = 8
        temporal_compression_ratio = 4
        tiling_enabled = False

        def enable_tiling(self) -> None:
            self.tiling_enabled = True

    class DummyVideoPipeline:
        def __init__(self) -> None:
            self.transformer = DummyDenoiser()
            self.vae = DummyVAE()
            self.vae_scale_factor_spatial = 8
            self.vae_scale_factor_temporal = 4

        def to(self, _device: Any) -> "DummyVideoPipeline":
            return self

        def __call__(
            self,
            prompt: str,
            width: int = 64,
            height: int = 64,
            num_frames: int = 9,
            num_inference_steps: int = 4,
            guidance_scale: float = 1.0,
            guidance_scale_2: float | None = None,
            generator: torch.Generator | None = None,
            output_type: str = "np",
        ) -> DummyOutput:
            assert isinstance(generator, torch.Generator)
            rng = torch.randn((num_frames, height, width, 3), generator=generator)
            arr = torch.sigmoid(rng).numpy().astype(np.float32)
            return DummyOutput(frames=arr)

    dummy_instance = DummyVideoPipeline()

    monkeypatch.setattr(
        "sd_task.task_runner.video_task.video_task._prepare_video_models",
        lambda args, cfg: {},
    )
    monkeypatch.setattr(
        "sd_task.task_runner.video_task.video_task.resolve_video_pipeline_spec",
        lambda args, cache_dir, local_files_only: ResolvedVideoPipelineSpec(
            kind="diffusion", pipeline_class=DummyVideoPipeline
        ),
    )
    monkeypatch.setattr(
        "sd_task.task_runner.video_task.video_task.resolve_video_dtype",
        lambda *a, **kw: ("bfloat16", torch.bfloat16),
    )
    monkeypatch.setattr(
        "sd_task.task_runner.video_task.video_task.load_video_pipeline",
        lambda *a, **kw: dummy_instance,
    )

    cfg = Config()
    cfg.local_files_only = True
    cache = MemoryModelCache()

    args = VideoTaskArgs.model_validate(
        {
            "version": "3.8.0",
            "base_model": {"name": "wan-ai/wan2.2-t2v-a14b-diffusers"},
            "dtype": "auto",
            "prompt": "A lantern floating across a calm river",
            "task_config": {
                "width": 64,
                "height": 64,
                "num_frames": 9,
                "fps": 16,
                "steps": 4,
                "seed": 123,
                "guidance_scale": 1.0,
            },
            "pipeline_args": {"guidance_scale_2": 1.0},
        }
    )

    out_dir_1 = str(tmp_path / "run1")
    out_dir_2 = str(tmp_path / "run2")
    res1 = run_video_task(args, output_dir=out_dir_1, config=cfg, model_cache=cache)
    res2 = run_video_task(args, output_dir=out_dir_2, config=cfg, model_cache=cache)

    assert dummy_instance.vae.tiling_enabled is True
    assert os.path.isfile(res1.output_path)
    assert os.path.isfile(res2.output_path)
    assert res1.execution_dtype == "bfloat16"
    assert res1.num_frames == 9
    assert res1.has_audio is False
    with av.open(res1.output_path) as c1, av.open(res2.output_path) as c2:
        p1 = [bytes(f.planes[0]) for f in c1.decode(c1.streams.video[0])]
        p2 = [bytes(f.planes[0]) for f in c2.decode(c2.streams.video[0])]
        assert p1 == p2

    bad_passthrough = args.model_copy(
        update={"pipeline_args": {"nonexistent_param": 123}}
    )
    with pytest.raises(TaskInvalid, match="nonexistent_param"):
        build_video_call_args(dummy_instance, bad_passthrough)
