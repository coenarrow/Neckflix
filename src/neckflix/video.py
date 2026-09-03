"""Streaming MKV decode and per-frame resize."""
from pathlib import Path

import av
import cv2
import numpy as np

_INTERPOLATION = {"rgb": cv2.INTER_AREA, "ir": cv2.INTER_AREA, "depth": cv2.INTER_NEAREST}


def decode_video(
    video_path: Path,
    kind: str,
    num_frames: int,
    resize: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    """Decode up to ``num_frames`` frames as ``(C, T, H, W)``, optionally resized.

    RGB decodes to uint8 (C=3); IR and depth decode to uint16 (C=1). Resize
    uses ``INTER_AREA`` for rgb/ir and ``INTER_NEAREST`` for depth so invalid-
    depth zeros are never interpolated into false distances.
    """
    if kind not in _INTERPOLATION:
        raise ValueError(f"Unknown stream kind: {kind!r}")
    interp = _INTERPOLATION[kind]

    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        fmt = stream.format.name
        if fmt not in ("bgr0", "gray16le"):
            raise ValueError(f"Unexpected stream format {fmt!r} in {video_path}")
        target = "rgb24" if fmt == "bgr0" else "gray16le"

        for frame in container.decode(stream):
            if len(frames) >= num_frames:
                break
            arr = frame.to_ndarray(format=target)  # (H, W, 3) uint8 | (H, W) uint16
            if resize is not None:
                h, w = resize
                arr = cv2.resize(arr, (w, h), interpolation=interp)  # cv2 wants (W, H)
            if arr.ndim == 2:
                arr = arr[..., None]  # (H, W) -> (H, W, 1)
            frames.append(arr)

    stacked = np.stack(frames)  # (T, H, W, C)
    return stacked.transpose(3, 0, 1, 2), fps  # (C, T, H, W)
