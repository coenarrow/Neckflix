"""Zarr store creation: schema and compression."""
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.codecs.numcodecs import Delta

from neckflix.events import EventData
from neckflix.traces import TRACE_UNITS, NativeTraces

_COMPRESSOR = BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")

NOMINAL_FPS = 30.0


def write_perspective_group(root: zarr.Group, name: str, fps: float | None) -> zarr.Group:
    """Create a perspective group and stamp its frame rate.

    ``fps`` is nominal and lives here rather than on each video group: a
    perspective's modalities are pixel-aligned and share a rate. ``None`` for
    the event perspective, which is asynchronous.
    """
    group = root.create_group(name)
    group.attrs["fps"] = fps
    return group


def init_store(store_path: Path, attrs: dict) -> zarr.Group:
    """Create (or wipe and recreate) a recording store and write root attrs."""
    root = zarr.open_group(store_path, mode="w")
    for key, value in attrs.items():
        root.attrs[key] = value
    return root


def write_video_group(
    modality_group: zarr.Group,
    frames: np.ndarray,
    timestamps_us: np.ndarray,
    chunk_size: int = 32,
) -> None:
    """Write ``video/data`` and the sibling ``timestamps_us/data``.

    Both live directly under the modality: the timestamps describe the
    modality's timeline, not the pixel array, so they are a sibling of
    ``video`` rather than a child of it (README, "Layout rules").
    """
    chunks = (frames.shape[0], chunk_size) + frames.shape[2:]  # (C, 32, H, W)
    video_group = modality_group.create_group("video")
    video_group.create_array(
        "data",
        data=frames,  # (C, T, H, W)
        chunks=chunks,
        compressors=[_COMPRESSOR],
        filters=[Delta(dtype=str(frames.dtype))],
    )
    write_timestamps(modality_group, timestamps_us)


def write_timestamps(parent: zarr.Group, timestamps_us: np.ndarray) -> None:
    """Write ``timestamps_us/data`` under ``parent``."""
    # asarray, not astype: the event timestamps are already int64 and are
    # hundreds of millions of samples long, so astype's unconditional copy
    # added a needless multi-GB allocation at the peak of the run.
    parent.create_group("timestamps_us").create_array(
        "data", data=np.asarray(timestamps_us, dtype=np.int64)
    )


def write_trace(
    parent: zarr.Group,
    name: str,
    values: np.ndarray,
    *,
    timestamps_us: np.ndarray | None = None,
    sample_rate: float | None = None,
) -> None:
    """Write ``{name}/data`` with its ``units`` attr under ``parent``.

    A frame-rate trace shares its modality's clock and needs nothing more. The
    native-rate traces under ``ev/ev`` do not -- events are asynchronous
    ``(N,)`` while the traces are a regular ``(M,)`` grid -- so those also
    carry their own ``timestamps_us`` sibling and a ``sample_rate`` attr. That
    is the one place the one-clock-per-modality rule is bent (README,
    "Layout rules").
    """
    group = parent.create_group(name)
    group.create_array("data", data=values.astype(np.float64))
    group.attrs["units"] = TRACE_UNITS[name]
    if timestamps_us is not None:
        group.attrs["sample_rate"] = sample_rate
        write_timestamps(group, timestamps_us)


def write_ev_perspective(
    root: zarr.Group,
    events: EventData,
    camera_offset_us: int,
    native: NativeTraces | None = None,
) -> None:
    """Write the event camera as an ``ev/ev/`` perspective/modality.

    ``x``/``y``/``p`` are trace-shaped ``(N,)`` arrays sharing the modality's
    event ``timestamps_us``. ``native`` adds the native-rate traces from
    ``trace_data.csv``, each on its own clock (see ``write_trace``).
    """
    perspective = write_perspective_group(root, "ev", fps=None)
    perspective.attrs["sensor_width"] = events.sensor_width
    perspective.attrs["sensor_height"] = events.sensor_height
    perspective.attrs["camera_offset_us"] = int(camera_offset_us)

    modality = perspective.create_group("ev")
    write_timestamps(modality, events.t)
    for name, values, units in (
        ("x", events.x, "px"),
        ("y", events.y, "px"),
        ("p", events.p, "arb"),
    ):
        group = modality.create_group(name)
        group.create_array("data", data=values)
        group.attrs["units"] = units

    if native is None:
        return
    for name, values in sorted(native.traces.items()):
        # The CSV header is the only place the real unit of a native trace is
        # stated; TRACE_UNITS is a hardcoded copy of what it said in 2026-09.
        # If a source file ever switched CVP to cmH2O, writing the map's value
        # anyway would mislabel a clinical trace with no visible symptom, so
        # a disagreement fails the recording instead.
        declared = native.units.get(name)
        if declared != TRACE_UNITS[name]:
            raise ValueError(
                f"trace_data.csv declares {name!r} in {declared!r} but "
                f"{TRACE_UNITS[name]!r} was expected. Refusing to write a "
                "mislabelled clinical trace: check the source file, then "
                "update TRACE_UNITS if the dataset really changed units."
            )
        write_trace(
            modality, name, values,
            timestamps_us=native.timestamps_us, sample_rate=native.sample_rate,
        )


def mark_complete(
    root: zarr.Group, modalities: list[str], perspectives: list[str]
) -> None:
    """Stamp a store as fully written (always the last write).

    Records what the run *asked for*, not what was written: a recording whose
    source lacks ir/depth would otherwise never satisfy a request for them and
    would re-decode rgb on every run forever.
    """
    root.attrs["complete"] = {
        "modalities": sorted(modalities),
        "perspectives": sorted(perspectives),
    }


def covers(
    store_path: Path,
    modalities: list[str],
    perspectives: list[str],
    resize: tuple[int, int] | None = None,
) -> bool:
    """True iff the store is complete AND spans everything this run wants.

    1.0.0 only checked that ``complete`` was present, so a store built with
    ``--modalities rgb`` silently satisfied a later run wanting all three.
    A bare ``True`` from such a store fails the subset test and is rebuilt.

    ``resize`` must match exactly, not be a subset: frames are already
    downsampled on disk, so a store built at 64x64 cannot serve a request for
    200x200 -- skipping it would hand back a resolution the user did not ask
    for. ``resized_to`` is a root attr in its own right; the ``complete`` dict
    stays exactly ``{modalities, perspectives}``.
    """
    if not Path(store_path).exists():
        return False
    try:
        attrs = dict(zarr.open_group(store_path, mode="r").attrs)
    except Exception:
        # Any unreadable store -- missing, truncated zarr.json from a killed
        # run, wrong format -- is by definition not a covering one, so say so
        # and let the caller rebuild rather than raising from a cache probe.
        return False
    done = attrs.get("complete")
    if not isinstance(done, dict):
        return False
    if attrs.get("resized_to") != (list(resize) if resize else None):
        return False
    return (
        set(modalities) <= set(done.get("modalities", []))
        and set(perspectives) <= set(done.get("perspectives", []))
    )
