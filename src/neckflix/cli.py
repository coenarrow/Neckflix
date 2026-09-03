"""neckflix-preprocess: align + resize Neckflix recordings into zarr stores."""
import argparse
import multiprocessing
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import zarr
from tqdm import tqdm

import neckflix
from neckflix.events import EcfPluginError, read_events
from neckflix.scan import (
    RecordingInfo, discover_recordings, read_dataset_info, scan_recording,
)
from neckflix.traces import (
    AlignedStream, align_stream, read_native_traces, read_sample_rate,
    read_stream_metadata,
)
from neckflix.video import decode_video
from neckflix.writer import (
    NOMINAL_FPS,
    covers,
    init_store,
    mark_complete,
    write_ev_perspective,
    write_perspective_group,
    write_trace,
    write_video_group,
)

VIDEO_MODALITIES = ("rgb", "ir", "depth")
VIDEO_PERSPECTIVES = ("1", "2")


@dataclass
class Job:
    """One recording's work order (picklable for multiprocessing)."""
    recording: RecordingInfo
    output_dir: Path
    resize: tuple[int, int] | None
    modalities: list[str]
    perspectives: list[str]
    overwrite: bool


@dataclass
class RecordingResult:
    name: str
    status: str  # "cached" | "skipped" | "failed"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    detail: str | None = None  # short content summary for the live status line


def _align_streams(
    job: Job, result: RecordingResult
) -> dict[tuple[str, str], AlignedStream]:
    """Align each video stream this run needs, from container metadata only.

    A stream is needed when it is being written, and every stream is needed
    when events are: the events window is a property of the recording, not of
    what this run asked to write, so ``--perspectives ev`` alone still trims
    events to the full video span.
    """
    rec = job.recording
    need_all = "ev" in job.perspectives and rec.events_path is not None
    aligned: dict[tuple[str, str], AlignedStream] = {}
    for perspective in sorted(rec.perspectives):
        for kind in VIDEO_MODALITIES:
            info = rec.perspectives[perspective].get(kind)
            wanted = perspective in job.perspectives and kind in job.modalities
            if info is None or not (wanted or need_all):
                continue
            timestamps, traces = read_stream_metadata(info.video_path)
            stream = align_stream(info.num_frames, timestamps, traces)
            for name, count in stream.interior_nans().items():
                result.warnings.append(
                    f"{perspective}/{kind}: {count} interior NaN sample(s) in "
                    f"{name}; kept as-is"
                )
            aligned[(perspective, kind)] = stream
    return aligned


def _write_video(
    root: zarr.Group,
    job: Job,
    aligned: dict[tuple[str, str], AlignedStream],
    result: RecordingResult,
) -> tuple[int, int]:
    """Decode, resize and write the requested video modalities.

    Returns ``(modalities written, frames written)``.
    """
    rec = job.recording
    groups: dict[str, zarr.Group] = {}  # perspective groups, created on first write
    modalities_written = frames_written = 0
    for (perspective, kind), stream in aligned.items():
        if perspective not in job.perspectives or kind not in job.modalities:
            continue
        info = rec.perspectives[perspective][kind]
        frames, fps = decode_video(
            info.video_path, kind, num_frames=stream.num_frames, resize=job.resize,
        )  # (C, T, H, W)
        if round(fps) != int(NOMINAL_FPS):
            result.warnings.append(
                f"{perspective}/{kind}: container rate {fps:.4f} is not "
                f"{NOMINAL_FPS:g}; keeping nominal fps"
            )
        decoded = frames.shape[1]
        if decoded < stream.num_frames:
            result.warnings.append(
                f"{perspective}/{kind}: decoded {decoded} frames < "
                f"{stream.num_frames} frames from metadata; truncating "
                "timestamps/traces to the decoded length"
            )
            stream = stream.truncated(decoded)
        if perspective not in groups:
            groups[perspective] = write_perspective_group(
                root, perspective, fps=NOMINAL_FPS
            )
        modality_group = groups[perspective].create_group(kind)
        write_video_group(modality_group, frames, timestamps_us=stream.timestamps_us)
        for name, values in stream.traces.items():
            write_trace(modality_group, name, values)
        modalities_written += 1
        frames_written += stream.num_frames
    return modalities_written, frames_written


def _write_events(
    root: zarr.Group,
    rec: RecordingInfo,
    aligned: dict[tuple[str, str], AlignedStream],
    result: RecordingResult,
) -> int | None:
    """Write the ``ev`` perspective; returns the event count, or None if skipped."""
    if rec.event_camera_offset_us is None:
        result.warnings.append("events skipped: no video_start_end_times.csv offset")
        return None
    if not aligned:
        # No recording in the dataset has EV.hdf5 without video, so this is
        # unreachable today -- but events are trimmed to the video window, and
        # without one there is nothing to trim to.
        result.warnings.append(
            "events skipped: recording has no video to define a window"
        )
        return None
    window = (
        min(int(s.timestamps_us[0]) for s in aligned.values()),
        max(int(s.timestamps_us[-1]) for s in aligned.values()),
    )
    events = read_events(
        rec.events_path, camera_offset_us=rec.event_camera_offset_us, window_us=window,
    )
    native = None
    if rec.trace_csv_path is not None:
        native = read_native_traces(rec.trace_csv_path)
        if rec.sample_rate_path is not None:
            native.sample_rate = read_sample_rate(rec.sample_rate_path)
            # sample_rate.json wins; the CSV column is "N/A" for the 3
            # recordings that have no traces at all, so it cannot be primary.
            # They agree on all 329 others.
            if (rec.csv_sample_rate is not None
                    and rec.csv_sample_rate != native.sample_rate):
                result.warnings.append(
                    f"sample_rate.json says {native.sample_rate:g} but "
                    f"dataset_info.csv says {rec.csv_sample_rate:g}; "
                    "using sample_rate.json"
                )
    write_ev_perspective(root, events, rec.event_camera_offset_us, native)
    return len(events.t)


def process_recording(job: Job) -> RecordingResult:
    """Align, decode/resize, and write one recording's zarr store."""
    rec = job.recording
    store_path = job.output_dir / f"{rec.name}.zarr"
    result = RecordingResult(name=rec.name, status="cached")
    try:
        if not job.overwrite and covers(
            store_path, job.modalities, job.perspectives, job.resize
        ):
            result.status = "skipped"
            return result

        root = init_store(
            store_path,
            {
                **rec.attrs,
                "source_resolution": rec.source_resolution,
                "resized_to": list(job.resize) if job.resize else None,
                "tool_version": neckflix.__version__,
            },
        )
        aligned = _align_streams(job, result)
        modalities_written, frames_written = _write_video(root, job, aligned, result)
        num_events = None
        if "ev" in job.perspectives and rec.events_path is not None:
            num_events = _write_events(root, rec, aligned, result)

        result.detail = f"{modalities_written} modalities, {frames_written} frames"
        if num_events is not None:
            result.detail += f", {num_events:,} events"
        mark_complete(root, job.modalities, job.perspectives)
    except EcfPluginError as exc:
        # A missing codec is a property of the environment, not of the
        # recording, so the "absent source files are not errors" rule does not
        # apply. Downgrading it to a warning would let mark_complete() stamp
        # the store as covering `ev` with no ev/ group ever written -- a
        # poisoned store that every later run skips and only --overwrite can
        # repair.
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
    except Exception:
        result.status = "failed"
        result.error = traceback.format_exc()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neckflix-preprocess",
        description="Temporally align (and optionally resize) Neckflix "
        "recordings into zarr stores.",
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Dataset root containing data/P*/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"),
                        default=None, help="Target frame size (default: native)")
    parser.add_argument("--recordings", nargs="+", default=None,
                        help="Recording names to process (default: all)")
    parser.add_argument("--modalities", nargs="+", default=list(VIDEO_MODALITIES),
                        choices=list(VIDEO_MODALITIES),
                        help="Video modalities to write (default: all)")
    parser.add_argument("--perspectives", nargs="+",
                        default=list(VIDEO_PERSPECTIVES) + ["ev"],
                        choices=list(VIDEO_PERSPECTIVES) + ["ev"],
                        help="Perspectives to write; 'ev' is the event camera "
                             "(default: all)")
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="Recordings processed in parallel (default: %(default)s). Peak "
             "RSS is ~12 GB per worker on the largest event recordings, so "
             "allow ~12 GB of RAM per worker.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--version", action="version", version=neckflix.__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        info_rows = read_dataset_info(args.input_dir)
        available = discover_recordings(args.input_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1
    names = args.recordings if args.recordings is not None else available
    if not names:
        print("No recordings found.")
        return 1

    # Probe each recording separately so a corrupt file in one recording
    # can't abort discovery of the rest.
    print(
        f"Found {len(names)} recording{'s' if len(names) != 1 else ''}; "
        "scanning metadata...",
        flush=True,
    )
    data_dir = args.input_dir / "data"
    jobs: list[Job] = []
    scan_failures: list[RecordingResult] = []
    for name in tqdm(names, desc="scanning", unit="rec", leave=False):
        rec_dir = data_dir / name
        if not rec_dir.is_dir():
            tqdm.write(f"WARNING: recording not found: {name}")
            continue
        try:
            rec = scan_recording(rec_dir, info_rows)
        except Exception:
            scan_failures.append(
                RecordingResult(name=name, status="failed", error=traceback.format_exc())
            )
            continue
        jobs.append(
            Job(recording=rec, output_dir=args.output_dir,
                resize=tuple(args.resize) if args.resize else None,
                modalities=args.modalities, perspectives=args.perspectives,
                overwrite=args.overwrite)
        )

    if not jobs and not scan_failures:
        # Every requested name (or, when unfiltered, every data/P* entry)
        # matched nothing scannable -- e.g. a typo'd --recordings value.
        print("No recordings found.")
        return 1

    for r in scan_failures:
        tqdm.write(f"FAILED (scan) {r.name}")

    def report(result: RecordingResult) -> None:
        """Print one live status line (plus warnings) as a recording finishes."""
        line = f"{result.status} {result.name}"
        if result.detail:
            line += f" ({result.detail})"
        tqdm.write(line)
        for w in result.warnings:
            tqdm.write(f"WARNING [{result.name}]: {w}")

    print(
        f"Processing {len(jobs)} recording{'s' if len(jobs) != 1 else ''} "
        f"with {max(args.num_workers, 1)} worker"
        f"{'s' if args.num_workers > 1 else ''}...",
        flush=True,
    )
    results: list[RecordingResult] = list(scan_failures)
    if args.num_workers <= 1:
        for job in tqdm(jobs, desc="recordings"):
            result = process_recording(job)
            report(result)
            results.append(result)
    else:
        with multiprocessing.Pool(args.num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_recording, jobs),
                total=len(jobs), desc="recordings",
            ):
                report(result)
                results.append(result)

    cached = [r for r in results if r.status == "cached"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    for r in failed:
        print(f"FAILED [{r.name}]:\n{r.error}")
    print(f"{len(cached)} cached, {len(skipped)} skipped, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
