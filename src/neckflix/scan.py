"""Discover release-format recordings and probe their metadata (no decode)."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

import av

ARTERIAL_SIDES = {"left", "right"}
ARTERIAL_SITES = {"radial", "brachial"}


def read_dataset_info(dataset_root: Path) -> dict[str, dict[str, str]]:
    """Load ``dataset_info.csv``, keyed by ``Recording_Directory``.

    Validates that the arterial-line side/site columns have not been swapped.
    These two were transposed in an earlier revision of the source file
    (``Radial`` in the side column, ``Right`` in the site column); the check is
    a regression guard, because a stale file would otherwise write wrong
    clinical metadata into every store without any visible symptom.
    """
    csv_path = Path(dataset_root) / "dataset_info.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No dataset_info.csv under {dataset_root}")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    bad_side = [
        r["Recording_Directory"] for r in rows
        if r["Arterial Line Insertion Side"].strip().lower() in ARTERIAL_SITES
    ]
    bad_site = [
        r["Recording_Directory"] for r in rows
        if r["Arterial Line Insertion Site"].strip().lower() in ARTERIAL_SIDES
    ]
    if bad_side or bad_site:
        raise ValueError(
            f"{csv_path}: arterial line side/site columns are swapped. "
            f"{len(bad_side)} row(s) have a site value in the side column "
            f"(e.g. {bad_side[:3]}); {len(bad_site)} row(s) have a side value "
            f"in the site column (e.g. {bad_site[:3]}). Fix the CSV before "
            "caching -- otherwise every store gets wrong line metadata."
        )
    return {r["Recording_Directory"]: r for r in rows}


POSTURE_BY_ANGLE = {"0": "supine", "45": "recumbent", "90": "sitting"}


def _s(value: str) -> str | None:
    """Strip a CSV cell; blank and ``N/A`` both mean missing."""
    v = value.strip()
    return None if v.lower() in ("", "n/a") else v


def _f(value: str) -> float | None:
    v = _s(value)
    return None if v is None else float(v)


def _i(value: str) -> int | None:
    v = _s(value)
    return None if v is None else int(float(v))


def _lower(value: str) -> str | None:
    v = _s(value)
    return None if v is None else v.lower()


def build_root_attrs(row: dict[str, str], name: str) -> dict:
    """Map one ``dataset_info.csv`` row to the store's root attributes.

    ``name`` is the recording directory name, used only to cross-check the
    posture token against the CSV's own ``Recording_ID``.
    """
    # "3_45_D" -> recording_id 3, angle 45. The directory name carries the same
    # tokens; disagreement means the CSV row and the directory have drifted
    # apart, which would silently mislabel posture, so it is fatal.
    rec_parts = row["Recording_ID"].split("_")
    angle = rec_parts[1]
    name_angle = name.split("_")[-2]
    if angle != name_angle:
        raise ValueError(
            f"{name}: posture token {name_angle!r} does not match "
            f"Recording_ID {row['Recording_ID']!r}"
        )
    if angle not in POSTURE_BY_ANGLE:
        raise ValueError(f"{name}: unknown posture angle {angle!r}")

    height_cm = _f(row["Height (cm)"])
    weight_kg = _f(row["Weight (kg)"])
    bmi = (
        round(weight_kg / (height_cm / 100) ** 2, 2)
        if height_cm and weight_kg else None
    )

    # The released dataset_info.csv (reconciled 2026-09-01) carries human-readable
    # labels, NOT the old -1 sentinel: "Not Visible" means the clinician looked but
    # could not see the pulse, "Not Assessed" means no clinician attended. Collapsing
    # both to null would lose a real distinction. That file also has no blank cells --
    # absence is spelled "N/A" -- and _s() maps blank and "N/A" alike to None.
    jvp_raw = _s(row["JVP Height Estimate (cm)"])
    if jvp_raw is None or jvp_raw == "Not Assessed":
        jvp_height, jvp_status = None, "no_clinician"
    elif jvp_raw == "Not Visible" or jvp_raw == "-1":
        # "-1" is the pre-2026-09-01 encoding. Kept deliberately: older copies
        # of this CSV still exist (including .bak files beside the canonical
        # one), and without this branch such a file would parse -1 as a real
        # measurement of -1 cm with status "measured" -- silently wrong rather
        # than a loud failure.
        jvp_height, jvp_status = None, "not_visualised"
    else:
        jvp_height, jvp_status = float(jvp_raw), "measured"

    return {
        "participant": _i(row["Participant_ID"]),
        "session": _i(row["Session_Number"]),
        "recording_id": _i(rec_parts[0]),
        "posture": POSTURE_BY_ANGLE[angle],

        "sex": _s(row["Sex"]),
        "age_years": _i(row["Age (years)"]),
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bmi": bmi,
        "neck_circumference_cm": {
            "lower": _f(row["Neck_Circumference_Lower (cm)"]),
            "mid": _f(row["Neck_Circumference_Mid (cm)"]),
            "upper": _f(row["Neck_Circumference_Upper (cm)"]),
        },
        "skin_tone": {
            "scale": "monk",
            "self": _i(row["Skin_Tone_Self"]),
            "recorder": _i(row["Skin_Tone_Recorder"]),
            "clinician": _i(row["Skin_Tone_Clinician"]),
        },
        "arrhythmia": _s(row["Arrhythmia"]) is not None,

        "central_line": {
            "site": _s(row["Central Line Insertion Site"]),
            "side": _lower(row["Central Line Insertion Side"]),
        },
        "arterial_line": {
            "site": _s(row["Arterial Line Insertion Site"]),
            "side": _lower(row["Arterial Line Insertion Side"]),
        },
        "neck_side_recorded": _lower(row["Neck Side Recorded"]),
        "bp_mmhg": {
            "systolic": _i(row["BP_Sys (mmHG)"]),
            "diastolic": _i(row["BP_Dia (mmHg)"]),
        },

        "jvp": {
            "height_cm": jvp_height,
            "status": jvp_status,
            "rater_job_title": _s(row["Job Title"]),
            "rater_years_practicing": _i(row["Years Practicing"]),
        },
        "heart_rate": {
            "beats_counted": _i(row["Beats_Counted"]),
            "beats_counted_unc": _i(row["Beats_Counted_Abs_Unc."]),
            "duration_s": _i(row["Duration"]),
            "average_bpm": _f(row["Average_BPM"]),
            "average_bpm_unc": _f(row["Average_BPM_Unc."]),
        },
    }


@dataclass
class StreamInfo:
    """One video stream of one perspective."""
    video_path: Path
    num_frames: int


@dataclass
class RecordingInfo:
    """Everything scan learned about one recording directory."""
    name: str
    attrs: dict
    perspectives: dict[str, dict[str, StreamInfo]] = field(default_factory=dict)
    events_path: Path | None = None
    event_camera_offset_us: int | None = None
    trace_csv_path: Path | None = None
    sample_rate_path: Path | None = None
    source_resolution: list[int] | None = None
    csv_sample_rate: float | None = None


def probe_video(video_path: Path) -> tuple[list[int], int]:
    """``([H, W], frame count)`` from container metadata: one open, no decode.

    The frame count tries the mkvmerge ``NUMBER_OF_FRAMES`` statistics tag,
    then ``stream.frames``, then ``duration * average_rate``.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        resolution = [int(stream.codec_context.height), int(stream.codec_context.width)]
        tag = stream.metadata.get("NUMBER_OF_FRAMES")
        if tag is not None:
            return resolution, int(tag)
        if stream.frames:
            return resolution, int(stream.frames)
        if container.duration and stream.average_rate:
            seconds = container.duration / 1_000_000
            return resolution, int(round(seconds * float(stream.average_rate)))
    raise ValueError(f"Cannot determine frame count for {video_path}")


def read_event_camera_offset(csv_path: Path) -> int | None:
    """``Camera_Offset`` (µs, int) of the ``EV.hdf5`` row, or None."""
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Video_Name"] == "EV.hdf5":
                return int(float(row["Camera_Offset"]))
    return None


def discover_recordings(input_dir: Path) -> list[str]:
    """Names of the ``data/P*`` recording directories (a listing, no probing)."""
    data_dir = Path(input_dir) / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No data/ directory under {input_dir}")
    return [p.name for p in sorted(data_dir.glob("P*")) if p.is_dir()]


def scan_recording(rec_dir: Path, info_rows: dict[str, dict[str, str]]) -> RecordingInfo:
    """Probe one recording directory's non-RAW MKVs and sidecar files."""
    name = rec_dir.name
    row = info_rows.get(name)
    if row is None:
        raise ValueError(
            f"{name}: no row in dataset_info.csv; cannot build root attributes"
        )
    rec = RecordingInfo(name=name, attrs=build_root_attrs(row, name))
    for video_path in sorted(rec_dir.glob("K*.mkv")):
        if "RAW" in video_path.name:
            continue
        perspective = "1" if video_path.name.startswith("K1") else "2"
        kind = video_path.stem.split("_")[1].lower()  # rgb | ir | depth
        resolution, num_frames = probe_video(video_path)
        # Probed rather than hardcoded: 1.0.0 asserted [650, 650] as a
        # constant. Every stream in a recording is the same size, so the
        # first one seen settles it.
        if rec.source_resolution is None:
            rec.source_resolution = resolution
        rec.perspectives.setdefault(perspective, {})[kind] = StreamInfo(
            video_path=video_path, num_frames=num_frames
        )
    ev_path = rec_dir / "EV.hdf5"
    if ev_path.exists():
        rec.events_path = ev_path
        rec.event_camera_offset_us = read_event_camera_offset(
            rec_dir / "video_start_end_times.csv"
        )
    trace_csv = rec_dir / "trace_data.csv"
    if trace_csv.exists():
        rec.trace_csv_path = trace_csv
    sample_rate_json = rec_dir / "sample_rate.json"
    if sample_rate_json.exists():
        rec.sample_rate_path = sample_rate_json
    # _f, not float(): Sample_Rate is the literal "N/A" for the 3 recordings
    # with no traces, and float("N/A") raises.
    rec.csv_sample_rate = _f(row["Sample_Rate"])
    return rec
