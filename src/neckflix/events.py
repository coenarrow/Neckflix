"""Event-camera reading: ECF decode, timebase mapping, temporal trim."""
from dataclasses import dataclass
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers bundled codecs; ECF needs HDF5_PLUGIN_PATH)
import numpy as np


class EcfPluginError(RuntimeError):
    """The ECF HDF5 codec plugin is not available in this environment."""


@dataclass
class EventData:
    """Columnar event stream in the common timebase."""
    x: np.ndarray  # (N,) uint16
    y: np.ndarray  # (N,) uint16
    p: np.ndarray  # (N,) int8
    t: np.ndarray  # (N,) int64, common timebase µs
    sensor_width: int
    sensor_height: int


def read_events(
    ev_path: Path, camera_offset_us: int, window_us: tuple[int, int]
) -> EventData:
    """Read CD/events, shift ``t`` by ``camera_offset_us``, trim to ``window_us``."""
    with h5py.File(ev_path, "r") as f:
        width, height = (int(v) for v in f.attrs["geometry"].split("x"))
        dataset = f["CD/events"]
        try:
            raw = dataset[:]  # compound (N,) — decodes via ECF plugin if compressed
        except OSError as exc:
            raise EcfPluginError(
                f"Cannot decode {ev_path}: the ECF HDF5 plugin is unavailable. "
                "Run inside the neckflix container, or build the plugin with "
                "build_scripts/build_plugin.sh and set HDF5_PLUGIN_PATH."
            ) from exc

    t = raw["t"].astype(np.int64) + camera_offset_us  # (N,) common timebase
    lo, hi = window_us
    mask = (t >= lo) & (t <= hi)  # (N,)
    return EventData(
        x=raw["x"][mask].astype(np.uint16),
        y=raw["y"][mask].astype(np.uint16),
        p=raw["p"][mask].astype(np.int8),
        t=t[mask],
        sensor_width=width,
        sensor_height=height,
    )
