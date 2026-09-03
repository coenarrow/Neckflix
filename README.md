# Neckflix

Preprocessing companion tool for the public **Neckflix dataset** release.
Neckflix recordings pair neck-mounted Kinect Azure video (RGB, IR, depth)
with synchronized physiological traces (ECG, CVP, ABP) and, optionally, an
event camera. This tool consumes the dataset's release-format recordings and
produces temporally aligned, optionally resized, [zarr](https://zarr.dev/)
stores — nothing more.

It is a **neutral dataset companion**: align + resize only. There is no
frame normalization, no trace filtering, and no windowing/chunking into
training samples — those are choices for downstream consumers (such as
[CardioHydra](https://github.com/coenarrow/CardioHydra)) to make.

The tool ships as a Docker image so you never have to build the HDF5 ECF
codec plugin (required to decode the event camera's `EV.hdf5` files)
yourself.

## Quick start (Docker)

Pre-built multi-arch images (amd64 + arm64) are published to GHCR:

```bash
docker pull ghcr.io/coenarrow/neckflix:latest

docker run --rm -t -v $DATA:/data:ro -v $OUT:/out ghcr.io/coenarrow/neckflix:latest \
  --input-dir /data --output-dir /out --resize 200 200
```

`$DATA` is the dataset root (it contains a `data/P*/` directory); `$OUT` is
where the `.zarr` stores are written. Drop `--resize` to keep native
650x650 frames. The `-t` flag is optional but makes the progress bars
render smoothly. Pin a version tag (e.g. `:1.0.0`) instead of `:latest`
when reproducibility matters.

Interrupted runs are safe to re-run with the same command: stores already
marked complete are skipped, and a store interrupted mid-write is detected
as partial and rebuilt.

To build the image yourself instead:

```bash
docker build -f docker/Dockerfile -t neckflix .
```

## Native (non-Docker) setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This is enough to process video and trace streams. Event-camera decoding
additionally needs the ECF HDF5 plugin built and on `HDF5_PLUGIN_PATH`:

```bash
bash build_scripts/build_plugin.sh
export HDF5_PLUGIN_PATH="$PWD/hdf5_plugin"
```

Without the plugin on `HDF5_PLUGIN_PATH`, any recording whose events were
requested **fails** — a missing codec is an environment fault, not missing
source data, and a store must never be marked complete for `ev` without an
`ev/` group in it. To process video and traces on a machine without the
plugin, deselect the event camera explicitly: `--perspectives 1 2`.

Then run the CLI directly:

```bash
uv run neckflix-preprocess --input-dir /data/Neckflix --output-dir /output --resize 200 200
```

## CLI reference

```text
neckflix-preprocess \
  --input-dir /data/Neckflix \        # required: root containing data/P*/
  --output-dir /output \              # required
  --resize 200 200 \                  # optional; default native 650x650
  --recordings P030_S01_R1_0_D ... \  # optional filter; default all
  --modalities rgb ir depth \         # optional; default all
  --perspectives 1 2 ev \             # optional; default all
  --num-workers 2 \                   # process-parallel over recordings (~12 GB RAM each)
  --overwrite                         # default: skip already-complete stores
```

| Flag | Required | Default | Notes |
| ---- | -------- | ------- | ----- |
| `--input-dir` | yes | — | Dataset root containing `data/P*/` |
| `--output-dir` | yes | — | Where `{recording}.zarr` stores are written |
| `--resize H W` | no | native (650x650) | Target frame size; RGB/IR use area interpolation, depth uses nearest-neighbor |
| `--recordings` | no | all found under `data/` | One or more recording names to process |
| `--modalities` | no | all | Video modalities to write: `rgb`, `ir`, `depth` |
| `--perspectives` | no | all | Perspectives to write: `1`, `2`, `ev` (event camera) |
| `--num-workers` | no | `2` | Recordings processed in parallel (one process per recording). Budget **~12 GB of RAM per worker**: the largest event recordings hold ~350 M events, and one worker peaks near 12 GB decoding them. A worker killed by the OOM killer makes the run hang with the progress bar one short rather than fail. |
| `--overwrite` | no | off | Rebuild stores that are already marked complete |
| `--version` | no | — | Print the installed version and exit |

**What a run looks like**: the tool first scans every recording's metadata
(a progress bar; on a slow or external drive this phase can take a few
minutes before any store is written), then processes recordings in
parallel, printing one status line per recording as it completes:

```text
Found 142 recordings; scanning metadata...
Processing 142 recordings with 4 workers...
cached P001_S01_R1_0_D (6 modalities, 5388 frames, 19,205,791 events)
cached P001_S01_R2_0_N (6 modalities, 5391 frames, 18,441,203 events)
WARNING [P002_S01_R1_0_D]: events skipped: no video_start_end_times.csv offset
...
142 cached, 0 skipped, 0 failed
```

Per-recording warnings (e.g. skipped events) print live next to their
recording; the traceback for any failure prints in the final summary.

**Exit status**: `0` if every requested recording was cached or skipped;
`1` if any recording failed, or if no recordings were found at all. The
summary line (`N cached, N skipped, N failed`) is always printed. A failing
recording does not stop the run — failures are isolated per recording.

## Input: release recording format

`{input_dir}/data/P{NNN}_S{NN}_R{n}_{angle}_{mode}/` — where `angle` is the bed
inclination (`0`/`45`/`90`, stored as `posture` = supine/recumbent/sitting) and
`mode` is the Kinect depth-camera mode (`D` = depth on; `N` = RGB only, no IR or
depth files).

`{input_dir}/dataset_info.csv` supplies every root attribute, joined on
`Recording_Directory`.

| File | Notes |
| ---- | ----- |
| `K{1,2}_{RGB,IR,DEPTH}.mkv` | 650x650; `bgr0` (RGB) / `gray16le` (IR, depth). Container metadata embeds `TIMESTAMPS_US` (per-frame, common timebase, int µs) and `ABP`/`CVP`/`ECG` traces as JSON arrays already interpolated to frame timestamps (may have NaN tails). `*_RAW.mkv` files are ignored. |
| `EV.hdf5` | Prophesee EVT3, 640x480. `CD/events` compound `(x:u2, y:u2, p:i2, t:i8)`, `t` in µs on the event camera's own clock. ECF-compressed → needs the plugin. |
| `video_start_end_times.csv` | Per-file rows with `Camera_Offset` etc. The `EV.hdf5` row's `Camera_Offset` maps event `t` into the common timebase. |
| `trace_data.csv` | Native-rate traces (`Time (s)` on the common timebase, plus `ABP`/`CVP`/`ECG` columns **in a non-fixed order** — match by header name). Absent for 3 recordings. |
| `sample_rate.json` | `{"sample_rate": 20000.0}` — 20 kHz for 266 recordings, 10 kHz for 63. |

Any of the MKVs, traces, or `EV.hdf5` may be absent for a given recording;
absence produces an output store without those groups, not an error.

## Output: zarr schema

One store per recording: `{output_dir}/{recording}.zarr`, following the
[layout rules](#layout-rules) below. The `ev` perspective is a
Neckflix-specific extension of that layout.

```text
P001_S01_R3_45_D.zarr
|-- attrs: participant, posture, jvp, heart_rate, ...  (see "Root attributes")
|-- 1/                                     attrs: fps=30
|   |-- rgb/
|   |   |-- timestamps_us/data  (T,) int64        common timebase
|   |   |-- video/data          (3,T,H,W) uint8   Delta + blosc-zstd-9 bitshuffle
|   |   |-- cvp/data            (T,) f64          attrs: units="mmHg"
|   |   `-- ecg/data            (T,) f64          attrs: units="mV"
|   |-- ir/                     video/data (1,T,H,W) uint16, otherwise identical
|   `-- depth/                  video/data (1,T,H,W) uint16, otherwise identical
|-- 2/                                     attrs: fps=30
`-- ev/    attrs: fps=null, sensor_width=640, sensor_height=480, camera_offset_us
    `-- ev/
        |-- timestamps_us/data  (N,) int64        event times, common timebase
        |-- x/data  (N,) uint16    attrs: units="px"
        |-- y/data  (N,) uint16    attrs: units="px"
        |-- p/data  (N,) int8      attrs: units="arb"
        |-- cvp/                   attrs: units="mmHg", sample_rate=20000
        |   |-- data               (M,) f64
        |   `-- timestamps_us/data (M,) int64
        `-- ecg/                   attrs: units="mV", sample_rate=20000
            |-- data               (M,) f64
            `-- timestamps_us/data (M,) int64
```

### Layout rules

The layout is the perspective/modality convention shared with
[CardioHydra](https://github.com/coenarrow/CardioHydra), which reads these
stores; the `ev` perspective extends it.

- **Perspective, then modality.** A perspective is one camera viewpoint: every
  frame in it is pixel-aligned (IR and depth were reprojected into the RGB
  camera at capture). Each modality under it came from its own sensor, so each
  carries its own `timestamps_us`, and every modality in a perspective carries
  the same set of traces.
- **`video/data` is `(C, T, H, W)`** and `timestamps_us/data` is `(T,)` on the
  common timebase. Both sit directly under the modality: the timestamps
  describe the modality's timeline, not the pixel array.
- **Every trace group carries a `units` attr** (`mmHg`, `mV`; `px` and `arb`
  for the event coordinates and polarity).
- **Root attrs carry `participant`**, which downstream loaders split on.
- **The `ev` perspective bends one rule.** An event camera is asynchronous, so
  it is its own perspective (it is not pixel-aligned to the Kinects) with a
  single `ev` modality, `fps=null`, and the sensor size and clock offset as
  attrs. `x`/`y`/`p` share the modality's event `timestamps_us`, but the
  native-rate traces are a regular grid and do not, so each of those carries
  its own `timestamps_us` sibling and a `sample_rate` attr. This is the only
  place a trace group has its own clock.

### Root attributes

These come from `dataset_info.csv`, joined on `Recording_Directory`; missing
values are written as `null`. `source_resolution`, `resized_to`,
`tool_version` and `complete` are added by the tool itself rather than read
from the CSV.

```python
{
  "participant": 1,                       # int  (required: downstream splits on it)
  "session": 1,                           # int
  "recording_id": 3,                      # int  (leading token of "3_45_D")
  "posture": "recumbent",                 # 0->supine, 45->recumbent, 90->sitting

  "sex": "M",                             # verbatim "M" | "F"
  "age_years": 73,                        # int
  "weight_kg": 95.9,                      # float
  "height_cm": 178.0,                     # float | null  (6 rows blank)
  "bmi": 30.27,                           # DERIVED weight_kg/(height_cm/100)^2; null if height null
  "neck_circumference_cm": {"lower": 42.0, "mid": 41.5, "upper": 45.0},   # lower/upper null in 128 rows
  "skin_tone": {"scale": "monk", "self": 4, "recorder": 5, "clinician": 5},
  "arrhythmia": False,                    # bool: blank -> False, 1 -> True

  "central_line":  {"site": "Basilic", "side": "right"},
  "arterial_line": {"site": None, "side": None},                          # null in ~70% of rows
  "neck_side_recorded": "right",
  "bp_mmhg": {"systolic": 108, "diastolic": 78},                          # null in 189 rows

  "jvp": {"height_cm": 2.0,
          "status": "measured",           # "measured" | "not_visualised" | "no_clinician"
          "rater_job_title": "RMO",
          "rater_years_practicing": 2},
  "heart_rate": {"beats_counted": 54, "beats_counted_unc": 1, "duration_s": 30,
                 "average_bpm": 108.0, "average_bpm_unc": 2.0},

  "source_resolution": [650, 650],        # PROBED from the MKV, not hardcoded
  "resized_to": None,                     # [H, W] | null
  "tool_version": "1.0.0",
  "complete": {"modalities": ["rgb", "ir", "depth"],          # see 3.3
               "perspectives": ["1", "2", "ev"]},
}
```

`complete` on the root attrs is written last and records what the run
requested; a store found without it, or one that does not cover the current
`--modalities`/`--perspectives`, is treated as incomplete and rebuilt. Frames
are stored as plain pixel values — the Delta filter and blosc compression are
zarr codecs, applied and inverted transparently on read.

Each modality keeps its own frame count; `timestamps_us` (on the common
timebase) is the cross-modality bridge, and the frame-rate traces are
duplicated under every modality group. The `ev` perspective additionally
carries the native-rate (10 or 20 kHz) traces from `trace_data.csv`.

## Reading the output

```python
import zarr

root = zarr.open_group("out/P001_S01_R1_0_D.zarr", mode="r")
frames = root["1/rgb/video/data"][:]          # (3, T, H, W) pixel values
window = root["1/rgb/video/data"][:, 100:132] # random windows read directly
t      = root["1/rgb/timestamps_us/data"][:]  # (T,) us, common timebase
ecg    = root["1/rgb/ecg/data"][:]            # (T,) mV, frame-rate
ecg_hz = root["ev/ev/ecg/data"][:]            # (M,) mV, native rate
et     = root["ev/ev/timestamps_us/data"][:]  # (N,) us, event times
```

## Out of scope

Spatial/cross-camera registration, event→frame conversion, frame
normalization, trace filtering, training-sample windowing, and a GUI/viewer
are all deliberately excluded. This tool stays a neutral dataset
companion — align + resize only; everything model-specific belongs
downstream.
