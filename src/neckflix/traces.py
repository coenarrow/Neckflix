"""Trace vocabulary, embedded trace/timestamp extraction and per-stream alignment."""
import json
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

# The physiological traces the release carries, with the units it states for
# them (the dataset_info.csv column headers: "ECG (mV)", "CVP (mmHg)",
# "ABP (mmHg)"). Keys are the lowercase group names written to the store; the
# MKV tags carry the same names in upper case.
TRACE_UNITS = {"abp": "mmHg", "cvp": "mmHg", "ecg": "mV"}


@dataclass
class AlignedStream:
    """Traces and timestamps truncated to a stream's aligned length."""
    num_frames: int
    timestamps_us: np.ndarray  # (T,) int64
    traces: dict[str, np.ndarray]  # name -> (T,) float64

    def truncated(self, num_frames: int) -> "AlignedStream":
        """Copy cut to the first ``num_frames`` frames."""
        return AlignedStream(
            num_frames=num_frames,
            timestamps_us=self.timestamps_us[:num_frames],
            traces={name: v[:num_frames] for name, v in self.traces.items()},
        )

    def interior_nans(self) -> dict[str, int]:
        """NaN samples per trace within the aligned span (tails are already trimmed)."""
        counts = {name: int(np.isnan(v).sum()) for name, v in self.traces.items()}
        return {name: n for name, n in counts.items() if n}


@dataclass
class NativeTraces:
    """``trace_data.csv`` at its acquisition rate; one clock shared by every column."""
    timestamps_us: np.ndarray  # (M,) int64
    traces: dict[str, np.ndarray]  # name -> (M,) float64
    units: dict[str, str | None]  # name -> units parsed from the CSV header
    sample_rate: float | None = None  # from sample_rate.json; None when absent


def read_stream_metadata(video_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read TIMESTAMPS_US and embedded trace JSON arrays from MKV metadata."""
    with av.open(str(video_path)) as container:
        timestamps = np.asarray(
            json.loads(container.metadata["TIMESTAMPS_US"]), dtype=np.int64
        )  # (T_video,)
        traces = {
            name: np.asarray(json.loads(container.metadata[name.upper()]), dtype=np.float64)
            for name in TRACE_UNITS
            if name.upper() in container.metadata
        }
    return timestamps, traces


def trim_trailing_nans(values: np.ndarray) -> np.ndarray:
    """Drop the trailing all-NaN tail; interior NaNs are kept."""
    finite = np.flatnonzero(~np.isnan(values))
    if len(finite) == 0:
        return values[:0]
    return values[: finite[-1] + 1]


def align_stream(
    video_frames: int,
    timestamps_us: np.ndarray,
    traces: dict[str, np.ndarray],
) -> AlignedStream:
    """Truncate traces and timestamps to ``min(video frames, trace lengths)``."""
    if len(timestamps_us) < video_frames:
        raise ValueError(
            f"{len(timestamps_us)} timestamps < {video_frames} video frames"
        )
    trimmed = {name: trim_trailing_nans(v) for name, v in traces.items()}
    num_frames = min([video_frames] + [len(v) for v in trimmed.values()])
    return AlignedStream(
        num_frames=num_frames,
        timestamps_us=timestamps_us[:num_frames],  # (T,)
        traces={name: v[:num_frames] for name, v in trimmed.items()},  # (T,)
    )


def parse_trace_header(column: str) -> tuple[str, str | None]:
    """Split a ``trace_data.csv`` header cell into ``(name, units)``.

    ``"CVP (mmHg)"`` -> ``("cvp", "mmHg")``; ``"Time (s)"`` -> ``("time", "s")``.
    ``units`` is ``None`` when the cell carries no parenthetical at all.
    """
    name, _, rest = column.partition(" (")
    units = rest.rsplit(")", 1)[0].strip() if rest else None
    return name.strip().lower(), units


def read_native_traces(csv_path: Path) -> NativeTraces:
    """Read ``trace_data.csv`` at its native acquisition rate.

    Columns are matched **by header name**, never by position: the order is not
    stable across the dataset (ABP precedes CVP in 6 recordings and trails ECG
    in 2). A positional read would write ABP samples under the name ``cvp``
    with nothing to reveal the swap.

    The units parsed out of each header are kept rather than thrown away: the
    CSV is the only place the real unit of a clinical trace is stated, and the
    writer checks it against its own map so a source file that switched CVP to
    cmH2O cannot be written labelled mmHg.
    """
    with open(csv_path) as f:
        header = [c.strip() for c in f.readline().strip().split(",")]
    parsed = [parse_trace_header(c) for c in header]
    names = [name for name, _ in parsed]
    if names[0] != "time":
        raise ValueError(f"{csv_path}: expected a leading Time column, got {header[0]!r}")

    values = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)  # (M, C)
    if values.shape[1] != len(names):
        raise ValueError(
            f"{csv_path}: header has {len(names)} columns but data has "
            f"{values.shape[1]}"
        )

    return NativeTraces(
        timestamps_us=np.round(values[:, 0] * 1_000_000).astype(np.int64),  # (M,)
        traces={
            name: values[:, i].astype(np.float64)
            for i, name in enumerate(names)
            if name != "time"
        },
        units={name: unit for name, unit in parsed if name != "time"},
    )


def read_sample_rate(json_path: Path) -> float:
    """Read ``sample_rate.json``. Not constant: 20 kHz for 266 recordings, 10 kHz for 63."""
    with open(json_path) as f:
        return float(json.load(f)["sample_rate"])
