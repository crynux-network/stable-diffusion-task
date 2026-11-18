from __future__ import annotations

import math
import struct

import av
import numpy as np

MAX_VIDEO_SCORE_FRAMES = 8
DCT_GRID_SIZE = 32
DCT_LOW_FREQ_SIZE = 8
AUDIO_TIME_SEGMENTS = 32
AUDIO_FREQ_BANDS = 32
AUDIO_WINDOW_SIZE = 64
AUDIO_MAX_WINDOWS_PER_SEGMENT = 64
AUDIO_LOG_EPSILON = 1e-10


def _build_dct2d_low_basis() -> list[list[float]]:
    basis: list[list[float]] = []
    for u in range(DCT_LOW_FREQ_SIZE):
        scale = math.sqrt(1.0 / DCT_GRID_SIZE) if u == 0 else math.sqrt(2.0 / DCT_GRID_SIZE)
        row = [
            scale * math.cos(math.pi * float(2 * k + 1) * float(u) / float(2 * DCT_GRID_SIZE))
            for k in range(DCT_GRID_SIZE)
        ]
        basis.append(row)
    return basis


def _build_audio_hann_dct_basis() -> tuple[list[float], list[list[float]]]:
    hann = [
        0.5 * (1.0 - math.cos(2.0 * math.pi * float(n) / float(AUDIO_WINDOW_SIZE)))
        for n in range(AUDIO_WINDOW_SIZE)
    ]
    dct_basis: list[list[float]] = []
    for band in range(AUDIO_FREQ_BANDS):
        row = [
            math.cos(math.pi * float(2 * n + 1) * float(band) / float(2 * AUDIO_WINDOW_SIZE))
            for n in range(AUDIO_WINDOW_SIZE)
        ]
        dct_basis.append(row)
    return hann, dct_basis


_DCT2D_LOW_BASIS = _build_dct2d_low_basis()
_AUDIO_HANN, _AUDIO_DCT_BASIS = _build_audio_hann_dct_basis()


def sampled_frame_indices(num_frames: int, max_samples: int = MAX_VIDEO_SCORE_FRAMES) -> list[int]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    count = min(max_samples, num_frames)
    if count == 1:
        return [0]
    return [(i * (num_frames - 1)) // (count - 1) for i in range(count)]


def downsample_luma_32x32(luma: np.ndarray) -> list[list[float]]:
    if luma.ndim != 2:
        raise ValueError(f"Expected 2D luma plane, got shape {luma.shape}")
    height, width = int(luma.shape[0]), int(luma.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid luma plane dimensions: {luma.shape}")

    luma_i64 = np.asarray(luma, dtype=np.int64)
    grid: list[list[float]] = []
    for r in range(DCT_GRID_SIZE):
        y0 = (r * height) // DCT_GRID_SIZE
        y1 = ((r + 1) * height) // DCT_GRID_SIZE
        if y1 <= y0:
            y1 = min(y0 + 1, height)
        row: list[float] = []
        for c in range(DCT_GRID_SIZE):
            x0 = (c * width) // DCT_GRID_SIZE
            x1 = ((c + 1) * width) // DCT_GRID_SIZE
            if x1 <= x0:
                x1 = min(x0 + 1, width)
            cell_sum = int(luma_i64[y0:y1, x0:x1].sum())
            count = (y1 - y0) * (x1 - x0)
            row.append(float(cell_sum) / float(count))
        grid.append(row)
    return grid


def dct8x8_median_hash(grid: list[list[float]]) -> bytes:
    row_dct: list[list[float]] = [[0.0] * DCT_LOW_FREQ_SIZE for _ in range(DCT_GRID_SIZE)]
    for r in range(DCT_GRID_SIZE):
        grid_row = grid[r]
        for v in range(DCT_LOW_FREQ_SIZE):
            basis_v = _DCT2D_LOW_BASIS[v]
            acc = 0.0
            for c in range(DCT_GRID_SIZE):
                acc += grid_row[c] * basis_v[c]
            row_dct[r][v] = acc

    coeffs: list[float] = []
    for u in range(DCT_LOW_FREQ_SIZE):
        basis_u = _DCT2D_LOW_BASIS[u]
        for v in range(DCT_LOW_FREQ_SIZE):
            acc = 0.0
            for r in range(DCT_GRID_SIZE):
                acc += basis_u[r] * row_dct[r][v]
            coeffs.append(acc)

    sorted_coeffs = sorted(coeffs)
    median = (sorted_coeffs[31] + sorted_coeffs[32]) / 2.0

    bits = 0
    for idx, val in enumerate(coeffs):
        if val > median:
            bits |= 1 << (63 - idx)
    return struct.pack(">Q", bits)


def hash_luma_frame(luma: np.ndarray) -> bytes:
    return dct8x8_median_hash(downsample_luma_32x32(luma))


def audio_spectrogram_32x32(mono_samples: np.ndarray) -> list[list[float]]:
    samples = np.asarray(mono_samples, dtype=np.float64).reshape(-1)
    total = int(samples.shape[0])
    if total == 0:
        return [
            [math.log(AUDIO_LOG_EPSILON)] * AUDIO_FREQ_BANDS
            for _ in range(AUDIO_TIME_SEGMENTS)
        ]

    hann_arr = np.asarray(_AUDIO_HANN, dtype=np.float64)
    dct_mat = np.asarray(_AUDIO_DCT_BASIS, dtype=np.float64)

    matrix: list[list[float]] = []
    for seg_idx in range(AUDIO_TIME_SEGMENTS):
        start = (seg_idx * total) // AUDIO_TIME_SEGMENTS
        end = ((seg_idx + 1) * total) // AUDIO_TIME_SEGMENTS
        if end <= start:
            end = min(start + 1, total)
        segment = samples[start:end]
        seg_len = int(segment.shape[0])

        if seg_len <= AUDIO_WINDOW_SIZE:
            padded = np.zeros(AUDIO_WINDOW_SIZE, dtype=np.float64)
            padded[:seg_len] = segment
            windows = padded.reshape(1, AUDIO_WINDOW_SIZE)
        else:
            max_start = seg_len - AUDIO_WINDOW_SIZE
            num_win = min(
                AUDIO_MAX_WINDOWS_PER_SEGMENT,
                max(1, seg_len // AUDIO_WINDOW_SIZE),
            )
            if num_win == 1:
                starts = [0]
            else:
                starts = [
                    (w * max_start) // (num_win - 1) for w in range(num_win)
                ]
            windows = np.stack(
                [segment[pos : pos + AUDIO_WINDOW_SIZE] for pos in starts],
                axis=0,
            )

        windowed = windows * hann_arr[None, :]
        proj = windowed @ dct_mat.T
        power = np.mean(proj * proj, axis=0)
        row = [math.log(float(power[band]) + AUDIO_LOG_EPSILON) for band in range(AUDIO_FREQ_BANDS)]
        matrix.append(row)
    return matrix


def hash_audio_pcm(mono_samples: np.ndarray) -> bytes:
    return dct8x8_median_hash(audio_spectrogram_32x32(mono_samples))


def _extract_frame_luma(frame: av.VideoFrame) -> np.ndarray:
    gray_frame = frame.reformat(format="gray")
    plane = gray_frame.planes[0]
    buf = np.frombuffer(plane, dtype=np.uint8)
    stride = int(plane.line_size)
    height = int(gray_frame.height)
    width = int(gray_frame.width)
    rows = buf[: height * stride].reshape(height, stride)
    return np.array(rows[:, :width], dtype=np.uint8, copy=True)


def _decode_mono_audio(
    container: av.container.InputContainer,
    audio_stream: av.audio.stream.AudioStream,
) -> np.ndarray:
    sample_rate = int(audio_stream.codec_context.sample_rate or 0)
    if sample_rate <= 0:
        return np.zeros(0, dtype=np.float64)
    resampler = av.audio.resampler.AudioResampler(
        format="fltp",
        layout=audio_stream.layout,
        rate=sample_rate,
    )
    chunks: list[np.ndarray] = []
    for frame in container.decode(audio_stream):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray().astype(np.float64, copy=False)
            chunks.append(arr.mean(axis=0))
    for resampled in resampler.resample(None):
        arr = resampled.to_ndarray().astype(np.float64, copy=False)
        chunks.append(arr.mean(axis=0))
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def compute_video_score(
    mp4_path: str,
    *,
    expected_num_frames: int | None = None,
) -> bytes:
    """Decode a written 0.mp4 file and compute its canonical multi-block VSS score."""
    with av.open(mp4_path, mode="r") as container:
        if not container.streams.video:
            raise RuntimeError("Encoded MP4 contains no video stream")
        video_stream = container.streams.video[0]

        luma_frames: list[np.ndarray] = []
        for frame in container.decode(video_stream):
            luma_frames.append(_extract_frame_luma(frame))

        total_frames = len(luma_frames)
        if total_frames == 0:
            raise RuntimeError("Encoded MP4 contains zero decodable video frames")
        if expected_num_frames is not None and total_frames != expected_num_frames:
            raise RuntimeError(
                f"Decoded MP4 frame count {total_frames} does not match "
                f"expected num_frames {expected_num_frames}"
            )

        indices = sampled_frame_indices(total_frames)
        blocks = [hash_luma_frame(luma_frames[idx]) for idx in indices]

        if container.streams.audio:
            container.seek(0)
            mono = _decode_mono_audio(container, container.streams.audio[0])
            blocks.append(hash_audio_pcm(mono))

    return b"".join(blocks)
