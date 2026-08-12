# ScreamFinder

Analyzes audio and video files for audible male and female vocalizations and generates an interactive HTML report with a built-in player.

Each file gets a **Female %** (fraction of time with detectable female-range vocalizations), a **Male %**, and a composite **Score** whose weights you can adjust live with sliders in the report.

---

## Requirements

- **Python 3.11+** (or Python 3.8+ with `tomli` installed for config file support)
- **ffmpeg** and **ffprobe** in your PATH
- Python packages: `numpy`, `scipy`
- Optional for `--detector yamnet`: `tensorflow`, `tensorflow-hub`

---

## Installation

### 1. Install ffmpeg

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Linux (apt):**
```bash
sudo apt install ffmpeg
```

**Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### 2. Install Python dependencies

Use a virtual environment (recommended on macOS with Homebrew Python, which blocks `pip install` into the system interpreter):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you prefer a global install and your Python allows it:

```bash
pip install numpy scipy
```

For the YAMNet detector path:

```bash
pip install -r requirements-yamnet.txt
```

If you are on Python 3.8–3.10 and want config file support, also install:

```bash
pip install tomli
```

### 3. Get the script

```bash
git clone <repo-url>
cd screamfinder
```

Run with the venv activated (`python3 screamfinder.py …`), or invoke the interpreter explicitly: `.venv/bin/python screamfinder.py …`.

---

## Quick Start

```bash
# Analyze all media files in a directory (video and/or audio)
python3 screamfinder.py ~/Media/ -o report.html

# Analyze specific files (mix and match video and audio)
python3 screamfinder.py clip1.mp4 song.mp3 podcast.flac -o report.html

# Open the report
open report.html          # macOS
xdg-open report.html      # Linux
```

The report opens in any modern browser. Chrome can open the HTML file directly (`file://`). **Safari** blocks media that lives outside the report's folder when using `file://`, so serve it instead:

```bash
# After analysis
python3 screamfinder.py ~/Media/ -o report.html --serve

# Or serve an existing report (no re-analysis)
python3 screamfinder.py --serve-report report.html
```

That starts a local server at `http://127.0.0.1:8765/` (override with `--port`) and opens your browser.

---

## HTML Report

The report shows a sortable table with one row per file:

| Column | Description |
|---|---|
| **#** | Current rank (updates as you sort) |
| **Filename** | Click to open the built-in video player |
| **Duration** | Full file duration |
| **Female %** | Percentage of frames with sustained female-range vocalizations |
| **Male %** | Percentage of frames with sustained male-range vocalizations |
| **Score** | Weighted composite; adjust weights with the sliders at the top |

### Player

Click any filename to open a full-screen modal player with:

- Click or **Space** / **K** — play / pause
- **← →** — seek ±5 seconds; **Shift + ← →** — seek ±30 seconds
- **↑ ↓** — volume; **M** — mute; scroll wheel over the player — volume
- **F** or double-click — fullscreen
- **0–9** — jump to 0%–90% of the file
- **Auto next** — when enabled, the next visible row starts automatically when the current file ends
- **Esc** — close

Audio files (`.mp3`, `.flac`, etc.) play in the same modal: instead of a video frame they show a static placeholder with the filename. The **timeline** and **volume** slider sit in the bar below that artwork (they stay visible while audio plays—no auto-hide). Scroll the mouse wheel over the artwork or video area to change volume. A small ♪ icon next to the filename in the table indicates audio rows.

If a file format isn't supported by the browser (e.g. `.mpg`, `.avi`, `.wma`), an error overlay appears with the file path and a **Copy path** button so you can paste it into VLC, IINA, or another player.

Safari note: opening the HTML via `file://` often cannot play `.mp4` files stored on another volume or folder. Use `--serve` / `--serve-report` so the player loads media over `http://127.0.0.1`.

---

## Command-Line Reference

```
python3 screamfinder.py [options] PATH [PATH ...]
```

`PATH` can be a media file (video or audio) or a directory (searched recursively). Multiple paths are accepted.

### Core options

| Flag | Default | Description |
|---|---|---|
| `-o FILE`, `--output FILE` | `screamfinder.html` | Output HTML file |
| `--serve` | off | After writing the report, serve it on localhost and open a browser (needed for Safari when media is outside the report folder) |
| `--serve-report FILE` | off | Serve an existing HTML report on localhost (no re-analysis) |
| `--port N` | `8765` | Port for `--serve` / `--serve-report` |
| `--segments-json FILE` | off | Optional JSON export with per-file detected segments and timestamps |
| `--config FILE` | `screamfinder.toml` | TOML config file (CLI args override) |
| `--jobs N` | `4` | Parallel analysis workers |
| `--force` | off | Re-analyze every file, ignoring cache |
| `--no-cache` | off | Disable caching entirely |

### Detection tuning

| Flag | Default | Description |
|---|---|---|
| `--detector NAME` | `heuristic` | Detection backend. `heuristic` is the current streaming STFT detector and is the integration point future model-based detectors will share. |
| `--yamnet-model HANDLE` | `https://tfhub.dev/google/yamnet/1` | TensorFlow Hub handle or local SavedModel path used when `--detector yamnet` is selected |
| `--yamnet-score-threshold N` | `0.05` | Threshold for YAMNet weighted sex-vocalization segment activation |
| `--yamnet-label-debug-json FILE` | off | Optional JSON export with per-window YAMNet vocalization scores and top AudioSet labels |
| `--yamnet-top-k N` | `8` | Number of top AudioSet labels to store per window in the YAMNet debug export |
| `--yamnet-min-window-rms RMS` | `0.005` | Absolute RMS backstop for YAMNet windows before scoring |
| `--yamnet-context-rms-seconds SEC` | `12.0` | Amount of preceding audio context used by the adaptive YAMNet energy gate |
| `--yamnet-context-rms-ratio RATIO` | `0.0` | Require each YAMNet window to be at least this fraction of the recent context RMS reference |
| `-t N`, `--threshold N` | `4.0` | A frame is counted as vocal when its band energy exceeds this multiple of the per-band noise floor. Raise to reduce false positives; lower to catch quieter vocalizations. |
| `--female-freq LOW HIGH` | `500 2000` | Female vocalization frequency range (Hz) |
| `--male-freq LOW HIGH` | `80 200` | Male vocalization frequency range (Hz) |
| `--min-vocal-duration SEC` | `0.5` | Minimum continuous duration (seconds) to count as a vocalization; shorter bursts are ignored |
| `--noise-floor-pct PCT` | `10.0` | Percentile of per-band frame energy used as the noise floor reference. Lower = more sensitive; higher = better rejection of constant background noise |
| `--min-audio-rms RMS` | `0.005` | Silence gate: files quieter than this RMS level (≈ −46 dBFS) are scored 0%. Set to `0` to disable. |

### Clip analysis

| Flag | Default | Description |
|---|---|---|
| `--clip-duration DUR` | `0` (full file) | Analyze only the last DUR of each file. Accepts plain seconds (`300`) or `[h:]mm:ss` (`5:00`, `1:05:00`). The full file duration is still shown in the report. |

### Audio / STFT parameters

| Flag | Default | Description |
|---|---|---|
| `--sample-rate HZ` | `11025` | Audio sample rate for analysis. Nyquist = sample\_rate / 2 must exceed `--female-freq HIGH`. |
| `--n-fft N` | `1024` | STFT window size in samples (power of 2). Larger = finer frequency resolution, slower. |
| `--hop-length N` | `1024` | STFT hop size in samples. Larger = fewer frames per second, faster analysis. |

### Cache

Results are cached to `.screamfinder-cache.json` so re-running on the same files is nearly instant. The cache key includes the file path, modification time, size, and every analysis parameter, so changing any setting automatically re-analyzes affected files.

Analysis now streams decoded PCM from `ffmpeg` in chunks instead of loading entire files into one in-memory array first, so very long media files use much less RAM during analysis. If you run with `--jobs 1`, analysis stays in the main process instead of starting a worker pool.

When `--segments-json` is enabled, ScreamFinder also writes timestamped detected segments for each file. Those segment records are intended to be stable across detector backends so future `yamnet` or `panns` integrations can export through the same JSON shape.

---

## Configuration File

Copy and edit `screamfinder.toml` to set persistent defaults. CLI arguments always take precedence.

```toml
# screamfinder.toml

threshold        = 4.0
clip_duration    = 0        # 0 = full file
min_vocal_duration = 0.5

female_freq = [500.0, 2000.0]
male_freq   = [80.0, 200.0]

sample_rate = 11025
n_fft       = 1024
hop_length  = 1024

jobs  = 4
cache = ".screamfinder-cache.json"
detector = "heuristic"
segments_json = ""
yamnet_model = "https://tfhub.dev/google/yamnet/1"
yamnet_score_threshold = 0.05
yamnet_label_debug_json = ""
yamnet_top_k = 8
yamnet_min_window_rms = 0.005
yamnet_context_rms_seconds = 12.0
yamnet_context_rms_ratio = 0.0

min_audio_rms   = 0.005
noise_floor_pct = 10.0
```

By default, `screamfinder.py` looks for `screamfinder.toml` in the current directory. Use `--config /path/to/other.toml` to specify a different file.

---

## Supported Formats

**Video:** `.mp4` `.avi` `.mkv` `.mov` `.wmv` `.flv` `.webm` `.m4v` `.mpg` `.mpeg` `.ts` `.3gp` `.m2ts` `.mts` `.vob` `.divx`

**Audio:** `.mp3` `.wav` `.ogg` `.oga` `.opus` `.m4a` `.aac` `.flac`

Directories are searched recursively. Any format ffmpeg can decode will work for analysis, but only formats the browser supports natively (MP4/H.264, WebM, Ogg, MP3, WAV, FLAC, AAC) will play directly in the report. Other formats — or audio files with exotic codecs inside an `.m4a`/`.aac` container — show a path-copy overlay so you can open them in VLC, IINA, or another external player.

---

## How It Works

1. **Audio extraction** — ffmpeg decodes the audio track to mono PCM at `--sample-rate` Hz.
   The decoder output is consumed in streaming chunks, so long files do not need to fit in RAM all at once.
2. **STFT** — a Short-Time Fourier Transform (window = `--n-fft`, hop = `--hop-length`) produces per-frame frequency energy.
3. **Per-band noise floor** — for each frequency band (female or male), the `--noise-floor-pct`-th percentile of that band's frame energies is used as the noise floor. This adapts to whatever constant background noise (hiss, music, etc.) is present.
4. **Silence gate** — files whose whole-file RMS is below `--min-audio-rms` are scored 0% immediately.
5. **Detection** — a frame is counted as "vocal" when its band energy exceeds `threshold × noise_floor`.
6. **Sustained filter** — isolated bursts shorter than `--min-vocal-duration` seconds are discarded.
7. **Result** — `female_pct` and `male_pct` are the percentage of frames that passed all filters.

### YAMNet detector

When `--detector yamnet` is selected, ScreamFinder decodes audio at `16000 Hz` and runs Google's YAMNet AudioSet classifier over streaming chunks. It combines scream, moan, crying, and whimper-style evidence into a single weighted `Vocal %` metric for sex-vocalization detection, and the JSON export labels positive segments as `vocalization`.

Before scoring each YAMNet window, ScreamFinder applies an adaptive RMS energy gate. A window must clear:

- the absolute backstop `--yamnet-min-window-rms`
- and, when enabled, a relative threshold based on the preceding `--yamnet-context-rms-seconds` of audio

This helps suppress near-silent or very weak windows without forcing the same fixed RMS floor on every file. The adaptive part is optional and is disabled by default when `--yamnet-context-rms-ratio` is `0.0`.

### YAMNet label debug export

If `--yamnet-label-debug-json debug.json` is provided, ScreamFinder writes one entry per YAMNet analysis window with:

- `start` and `end`
- `audio_rms` and `audio_rms_threshold`
- `energy_gated`
- computed `vocalization_score`
- `top_labels`, containing the highest-scoring raw AudioSet labels for that window

This export is meant for tuning the combined vocalization weight map and threshold against your real samples.

### Sample annotation format

A spreadsheet is a good fit for review data. The minimum useful columns are:

- `filename`
- `start_sec`
- `end_sec`
- `label`

Recommended optional columns:

- `notes`
- `confidence`
- `split` (`train`, `dev`, `test`)

### Segment JSON

If `--segments-json out.json` is provided, the export includes one entry per analyzed file with:

- file path and name
- detector name
- overall percentages
- a `segments` array with `label`, `start`, `end`, `duration`, `frame_count`, `peak_ratio`, `avg_ratio`, and frequency bounds

---

## Tips

**Too many false positives (music, ambient noise detected as vocals):**
- Raise `--threshold` (try 5–8)
- Raise `--noise-floor-pct` (try 15–25)
- Raise `--min-vocal-duration` (try 1.0–2.0)

**Vocalizations missed:**
- Lower `--threshold` (try 2–3)
- Lower `--min-vocal-duration` (try 0.2)
- Lower `--min-audio-rms` or set to `0`

**Quiet files scored as 0%:**
- Lower or disable `--min-audio-rms` (`--min-audio-rms 0`)

**Only care about the climax of each file:**
- Use `--clip-duration 5:00` to analyze only the last 5 minutes (works for both audio and video)

**Speed up analysis of a large collection:**
- Raise `--jobs` to match your CPU core count
- The cache means subsequent runs (with the same settings) are nearly instant
