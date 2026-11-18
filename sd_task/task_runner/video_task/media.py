from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO

import av
import numpy as np
import torch
from PIL import Image

from sd_task.task_runner.image_task.errors import TaskInvalid


@dataclass(frozen=True)
class DecodedVideoInput:
    frames: list[Image.Image]
    fps: float
    audio: torch.Tensor | None
    audio_sample_rate: int | None


@dataclass(frozen=True)
class DecodedAudioInput:
    waveform: torch.Tensor
    sample_rate: int


def _split_data_url(data_url: str, expected_prefix: str, label: str) -> bytes:
    header, separator, payload = data_url.partition(",")
    if (
        separator == ""
        or not header.startswith(expected_prefix)
        or ";base64" not in header
    ):
        raise TaskInvalid(
            f"{label} inputs must be base64-encoded {expected_prefix}* data URLs"
        )
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise TaskInvalid(f"{label} input is not valid base64") from exc


def decode_image_input(data_url: str) -> Image.Image:
    raw = _split_data_url(data_url, "data:image/", "Image")
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            return image.convert("RGB")
    except Exception as exc:
        raise TaskInvalid("Image input is not a valid encoded image") from exc


def _decode_audio_stream(
    container: av.container.InputContainer,
    stream: av.audio.stream.AudioStream,
) -> tuple[torch.Tensor, int]:
    sample_rate = int(stream.codec_context.sample_rate or 0)
    if sample_rate <= 0:
        raise TaskInvalid("Audio stream has an invalid sample rate")
    resampler = av.audio.resampler.AudioResampler(
        format="fltp", layout=stream.layout, rate=sample_rate
    )
    chunks: list[torch.Tensor] = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            chunks.append(torch.from_numpy(resampled.to_ndarray()))
    for resampled in resampler.resample(None):
        chunks.append(torch.from_numpy(resampled.to_ndarray()))
    if not chunks:
        raise TaskInvalid("Audio stream contains no decodable audio samples")
    return torch.cat(chunks, dim=-1).to(torch.float32), sample_rate


def decode_video_input(data_url: str) -> DecodedVideoInput:
    raw = _split_data_url(data_url, "data:video/", "Video")
    try:
        with av.open(BytesIO(raw), mode="r") as container:
            if not container.streams.video:
                raise TaskInvalid("Video input contains no video stream")
            video_stream = container.streams.video[0]
            frames_np: list[np.ndarray] = []
            rotation = 0.0
            for frame in container.decode(video_stream):
                rotation = float(getattr(frame, "rotation", 0.0) or 0.0)
                frames_np.append(frame.to_ndarray(format="rgb24"))
            if not frames_np:
                raise TaskInvalid("Video input contains zero frames")
            rate = video_stream.average_rate or video_stream.guessed_rate
            fps = float(rate) if rate else 24.0
            if fps <= 0:
                fps = 24.0

            audio_tensor: torch.Tensor | None = None
            audio_rate: int | None = None
            if container.streams.audio:
                container.seek(0)
                audio_tensor, audio_rate = _decode_audio_stream(
                    container, container.streams.audio[0]
                )
    except TaskInvalid:
        raise
    except Exception as exc:
        raise TaskInvalid("Video input is not a valid encoded video") from exc

    turns = round(rotation / 90.0) % 4
    pil_frames: list[Image.Image] = []
    for arr in frames_np:
        if turns:
            arr = np.ascontiguousarray(np.rot90(arr, k=-turns, axes=(0, 1)))
        pil_frames.append(Image.fromarray(arr, mode="RGB"))

    return DecodedVideoInput(
        frames=pil_frames,
        fps=fps,
        audio=audio_tensor,
        audio_sample_rate=audio_rate,
    )


def decode_audio_input(data_url: str) -> DecodedAudioInput:
    raw = _split_data_url(data_url, "data:audio/", "Audio")
    try:
        with av.open(BytesIO(raw), mode="r") as container:
            if not container.streams.audio:
                raise TaskInvalid("Audio input contains no audio stream")
            waveform, sample_rate = _decode_audio_stream(
                container, container.streams.audio[0]
            )
    except TaskInvalid:
        raise
    except Exception as exc:
        raise TaskInvalid("Audio input is not a valid encoded audio clip") from exc

    return DecodedAudioInput(waveform=waveform, sample_rate=sample_rate)
