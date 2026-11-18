from __future__ import annotations

from fractions import Fraction
from typing import Any

import av
import numpy as np
import torch
from PIL import Image


def normalize_video_frames(frames: Any) -> np.ndarray:
    """Convert pipeline frame outputs into a contiguous uint8 RGB array (T, H, W, 3)."""
    if isinstance(frames, (list, tuple)):
        if len(frames) == 0:
            raise RuntimeError("The video pipeline returned zero frames")
        first = frames[0]
        if isinstance(first, (list, tuple)):
            frames = first
        elif isinstance(first, (np.ndarray, torch.Tensor)) and len(frames) == 1:
            candidate = first.detach().cpu().numpy() if isinstance(first, torch.Tensor) else first
            if candidate.ndim == 4:
                frames = candidate

    if isinstance(frames, (list, tuple)):
        if len(frames) == 0:
            raise RuntimeError("The video pipeline returned zero frames")
        if isinstance(frames[0], Image.Image):
            rgb_arrays = [np.asarray(img.convert("RGB"), dtype=np.uint8) for img in frames]
            return np.ascontiguousarray(np.stack(rgb_arrays, axis=0))
        frames = np.stack(
            [
                f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
                for f in frames
            ],
            axis=0,
        )

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().float().numpy()
    elif not isinstance(frames, np.ndarray):
        raise RuntimeError(f"Unsupported video frames output type: {type(frames)!r}")

    if frames.ndim == 5:
        if frames.shape[0] != 1:
            raise RuntimeError(f"Expected batch size 1 for video output, got {frames.shape}")
        frames = frames[0]

    if frames.ndim != 4:
        raise RuntimeError(f"Expected 4D video array (T, H, W, C) or (T, C, H, W), got shape {frames.shape}")

    if frames.shape[-1] != 3 and frames.shape[1] == 3:
        frames = np.transpose(frames, (0, 2, 3, 1))

    if frames.shape[-1] != 3:
        raise RuntimeError(f"Expected 3 RGB channels in video frames, got shape {frames.shape}")

    if frames.dtype != np.uint8:
        arr = frames.astype(np.float32, copy=False)
        if arr.min() < -0.01:
            arr = (arr + 1.0) * 0.5
        if arr.max() <= 1.05:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        frames = np.rint(arr).astype(np.uint8)

    return np.ascontiguousarray(frames)


def _normalize_stereo_int16_audio(audio: torch.Tensor | np.ndarray) -> np.ndarray:
    """Normalize an audio waveform into stereo packed int16 array of shape (1, 2 * num_samples)."""
    if isinstance(audio, torch.Tensor):
        samples = audio.detach().cpu().float().numpy()
    else:
        samples = np.asarray(audio, dtype=np.float32)

    while samples.ndim > 2 and samples.shape[0] == 1:
        samples = samples[0]

    if samples.ndim == 1:
        samples = np.stack([samples, samples], axis=-1)
    elif samples.ndim == 2:
        if samples.shape[0] in (1, 2) and samples.shape[1] > 2:
            samples = samples.T
        elif samples.shape[1] not in (1, 2) and samples.shape[0] in (1, 2):
            samples = samples.T
        if samples.shape[1] == 1:
            samples = np.repeat(samples, 2, axis=1)
        elif samples.shape[1] != 2:
            raise RuntimeError(f"Unsupported audio channel layout shape: {samples.shape}")
    else:
        raise RuntimeError(f"Unsupported audio tensor dimensions: {samples.shape}")

    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = np.rint(clipped * 32767.0).astype(np.int16)
    return np.ascontiguousarray(pcm16.reshape(1, -1))


def encode_canonical_mp4(
    frames_rgb: np.ndarray,
    *,
    fps: float,
    output_path: str,
    audio: torch.Tensor | np.ndarray | None = None,
    audio_sample_rate: int | None = None,
    crf: int = 18,
) -> tuple[int, int, int, bool]:
    """Deterministically encode uint8 RGB frames and optional audio to H.264/AAC MP4.

    Returns (num_frames, width, height, has_audio).
    """
    frames = normalize_video_frames(frames_rgb)
    num_frames, height, width, _ = frames.shape
    if num_frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions: {frames.shape}")
    if fps <= 0:
        raise RuntimeError(f"Invalid video fps: {fps}")

    has_audio = audio is not None
    if has_audio and (audio_sample_rate is None or int(audio_sample_rate) <= 0):
        raise RuntimeError("Positive audio_sample_rate is required when audio is present")

    rate = Fraction(float(fps)).limit_denominator(1000)
    container = av.open(output_path, mode="w")
    try:
        video_stream = container.add_stream("libx264", rate=rate)
        video_stream.width = int(width)
        video_stream.height = int(height)
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {
            "crf": str(int(crf)),
            "threads": "1",
        }

        audio_stream = None
        if has_audio:
            sample_rate = int(audio_sample_rate)  # type: ignore[arg-type]
            audio_stream = container.add_stream("aac", rate=sample_rate)
            audio_stream.layout = "stereo"

        for idx in range(num_frames):
            video_frame = av.VideoFrame.from_ndarray(frames[idx], format="rgb24")
            for packet in video_stream.encode(video_frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        if has_audio and audio_stream is not None:
            sample_rate = int(audio_sample_rate)  # type: ignore[arg-type]
            packed_s16 = _normalize_stereo_int16_audio(audio)  # type: ignore[arg-type]
            if packed_s16.size > 0:
                frame_in = av.AudioFrame.from_ndarray(
                    packed_s16, format="s16", layout="stereo"
                )
                frame_in.sample_rate = sample_rate
                resampler = av.audio.resampler.AudioResampler(
                    format=audio_stream.codec_context.format or "fltp",
                    layout="stereo",
                    rate=sample_rate,
                )
                next_pts = 0
                for rframe in resampler.resample(frame_in):
                    if rframe.pts is None:
                        rframe.pts = next_pts
                    next_pts += rframe.samples
                    rframe.sample_rate = sample_rate
                    for packet in audio_stream.encode(rframe):
                        container.mux(packet)
                for rframe in resampler.resample(None):
                    if rframe.pts is None:
                        rframe.pts = next_pts
                    next_pts += rframe.samples
                    rframe.sample_rate = sample_rate
                    for packet in audio_stream.encode(rframe):
                        container.mux(packet)
                for packet in audio_stream.encode():
                    container.mux(packet)
    finally:
        container.close()

    return int(num_frames), int(width), int(height), has_audio
