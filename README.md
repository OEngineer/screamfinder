# ScreamFinder

Analyzes audio and video files for audible male and female vocalizations and generates an interactive HTML report with a built-in player.

Each file gets a **Female %** (fraction of time with detectable female-range vocalizations), a **Male %**, and a composite **Score** whose weights you can adjust live with sliders in the report.

---

## Requirements

- **Python 3.11+** (or Python 3.8+ with `tomli` installed for config file support)
- **ffmpeg** and **ffprobe** in your PATH
- Python packages: `numpy`, `scipy`

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

The report opens in any modern browser. No web server required.

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
- **Esc** — close

Audio files (`.mp3`, `.flac`, etc.) play in the same modal: instead of a video frame they show a static placeholder with the filename, but every keybinding and control above still applies. A small ♪ icon next to the filename in the table indicates audio rows.

If a file format isn't supported by the browser (e.g. `.mpg`, `.avi`, `.wma`), an error overlay appears with the file path and a **Copy path** button so you can paste it into VLC, IINA, or another player.

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
| `--config FILE` | `screamfinder.toml` | TOML config file (CLI args override) |
| `--jobs N` | `4` | Parallel analysis workers |
| `--force` | off | Re-analyze every file, ignoring cache |
| `--no-cache` | off | Disable caching entirely |

### Detection tuning

| Flag | Default | Description |
|---|---|---|
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
2. **STFT** — a Short-Time Fourier Transform (window = `--n-fft`, hop = `--hop-length`) produces per-frame frequency energy.
3. **Per-band noise floor** — for each frequency band (female or male), the `--noise-floor-pct`-th percentile of that band's frame energies is used as the noise floor. This adapts to whatever constant background noise (hiss, music, etc.) is present.
4. **Silence gate** — files whose whole-file RMS is below `--min-audio-rms` are scored 0% immediately.
5. **Detection** — a frame is counted as "vocal" when its band energy exceeds `threshold × noise_floor`.
6. **Sustained filter** — isolated bursts shorter than `--min-vocal-duration` seconds are discarded.
7. **Result** — `female_pct` and `male_pct` are the percentage of frames that passed all filters.

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
