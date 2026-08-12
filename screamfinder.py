#!/usr/bin/env python3
"""
screamfinder - Analyzes audio and video files for audible male/female vocalizations.

Detects and quantifies vocalizations in media files (video or audio) and
generates an interactive HTML report with a built-in player.

Requirements:
    pip install numpy scipy
    brew install ffmpeg  (or equivalent)

Usage:
    python3 screamfinder.py [options] <file_or_dir> [<file_or_dir> ...]

Examples:
    python3 screamfinder.py ~/Videos/ -o report.html
    python3 screamfinder.py clip1.mp4 song.mp3 --threshold 2.0
    python3 screamfinder.py ~/Media/ --female-freq 300 2500 --male-freq 80 600
    python3 screamfinder.py ~/Media/ -o report.html --serve
    python3 screamfinder.py --serve-report report.html --browser safari
"""

import argparse
import csv
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import label as nd_label

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv",
    ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".3gp",
    ".m2ts", ".mts", ".vob", ".divx",
})

AUDIO_EXTENSIONS = frozenset({
    ".mp3", ".wav", ".ogg", ".oga", ".opus",
    ".m4a", ".aac", ".flac",
})

MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
STREAM_CHUNK_SECONDS = 120.0
DETECTOR_CHOICES = ("heuristic", "yamnet")
ANALYSIS_CACHE_VERSION = 2
YAMNET_SAMPLE_RATE = 16000
YAMNET_PATCH_WINDOW_SECONDS = 0.96
YAMNET_PATCH_HOP_SECONDS = 0.48
YAMNET_CHUNK_SECONDS = 30.0
YAMNET_CHUNK_OVERLAP_SECONDS = 1.0
YAMNET_MIN_WINDOW_RMS = 0.005
YAMNET_CONTEXT_RMS_SECONDS = 12.0
YAMNET_CONTEXT_RMS_RATIO = 0.0
YAMNET_CONTEXT_RMS_PERCENTILE = 75.0
HOTSPOT_MERGE_GAP_SECONDS = 12.0
HOTSPOT_PADDING_SECONDS = 4.0
HOTSPOT_MIN_DURATION_SECONDS = 12.0
HOTSPOT_SEEK_PREROLL_SECONDS = 2.5
YAMNET_VOCALIZATION_WEIGHTS = {
    "Screaming": 1.00,
    "Wail, moan": 0.95,
    "Groan": 0.90,
    "Crying, sobbing": 0.90,
    "Whimper": 0.80,
    "Baby cry, infant cry": 0.55,
    "Gasp": 0.35,
    "Breathing": 0.18,
    "Pant": 0.20,
    "Grunt": 0.15,
    "Sigh": 0.10,
    "Yell": 0.50,
    "Shout": 0.45,
}
YAMNET_NEGATIVE_WEIGHTS = {
    "Speech": 0.20,
    "Conversation": 0.25,
    "Narration, monologue": 0.30,
    "Music": 1.00,
    "Background music": 1.10,
    "Whoop": 0.60,
    "Bellow": 0.70,
    "Crowd": 0.75,
    "Cheering": 0.75,
    "Children shouting": 0.60,
    "Hubbub, speech noise, speech babble": 0.60,
    "Dog": 0.55,
    "Domestic animals, pets": 0.45,
    "Animal": 0.30,
    "Whimper (dog)": 0.65,
    "Yip": 0.55,
    "Bow-wow": 0.55,
    "Howl": 0.50,
}


def media_kind(path: Path) -> str:
    """Return "audio" if path's suffix is a known audio extension, else "video"."""
    return "audio" if path.suffix.lower() in AUDIO_EXTENSIONS else "video"


def detector_metric_labels(detector: str) -> Tuple[str, Optional[str]]:
    if detector == "yamnet":
        return "Vocal %", None
    return "Female %", "Male %"


def _segment_strength(segment: Dict[str, object]) -> float:
    for key in ("peak_score", "avg_score", "peak_ratio", "avg_ratio"):
        value = segment.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _union_duration(intervals: List[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        total += max(0.0, cur_end - cur_start)
        cur_start, cur_end = start, end
    total += max(0.0, cur_end - cur_start)
    return total


def build_hotspots(
    segments: List[dict],
    duration: Optional[float],
    merge_gap: float = HOTSPOT_MERGE_GAP_SECONDS,
    padding: float = HOTSPOT_PADDING_SECONDS,
    min_duration: float = HOTSPOT_MIN_DURATION_SECONDS,
    seek_preroll: float = HOTSPOT_SEEK_PREROLL_SECONDS,
) -> List[dict]:
    if not segments:
        return []

    max_duration = float(duration) if duration is not None and duration > 0 else None
    ordered = sorted(
        (
            seg for seg in segments
            if float(seg.get("end", 0.0)) > float(seg.get("start", 0.0))
        ),
        key=lambda seg: (float(seg["start"]), float(seg["end"])),
    )
    if not ordered:
        return []

    clusters: List[List[dict]] = []
    current: List[dict] = [ordered[0]]
    current_end = float(ordered[0]["end"])
    for seg in ordered[1:]:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        if seg_start <= current_end + merge_gap:
            current.append(seg)
            current_end = max(current_end, seg_end)
            continue
        clusters.append(current)
        current = [seg]
        current_end = seg_end
    clusters.append(current)

    hotspots: List[dict] = []
    for cluster in clusters:
        raw_start = float(cluster[0]["start"])
        raw_end = max(float(seg["end"]) for seg in cluster)
        raw_span = max(0.0, raw_end - raw_start)
        if raw_span <= 0:
            continue

        interval_pairs = sorted((float(seg["start"]), float(seg["end"])) for seg in cluster)
        positive_duration = _union_duration(interval_pairs)
        density = positive_duration / raw_span if raw_span > 0 else 0.0
        peak_strength = max(_segment_strength(seg) for seg in cluster)
        labels = sorted({str(seg.get("label", "vocalization")) for seg in cluster})

        nav_start = max(0.0, raw_start - padding)
        nav_end = raw_end + padding
        if max_duration is not None:
            nav_end = min(max_duration, nav_end)
        nav_span = nav_end - nav_start
        if nav_span < min_duration:
            center = (raw_start + raw_end) / 2.0
            half = min_duration / 2.0
            nav_start = max(0.0, center - half)
            nav_end = center + half
            if max_duration is not None:
                if nav_end > max_duration:
                    nav_end = max_duration
                    nav_start = max(0.0, nav_end - min_duration)
                else:
                    nav_start = max(0.0, nav_start)
            nav_span = nav_end - nav_start

        seek_to = max(0.0, raw_start - seek_preroll)
        if max_duration is not None:
            seek_to = min(max_duration, seek_to)

        hotspots.append({
            "start": round(nav_start, 3),
            "end": round(nav_end, 3),
            "seek_to": round(seek_to, 3),
            "duration": round(max(0.0, nav_span), 3),
            "raw_start": round(raw_start, 3),
            "raw_end": round(raw_end, 3),
            "positive_duration": round(positive_duration, 3),
            "density": round(density, 4),
            "peak_strength": round(peak_strength, 4),
            "segment_count": len(cluster),
            "labels": labels,
        })
    return hotspots


def build_report_subtitle(file_count: int, args: argparse.Namespace) -> str:
    clip_info = f" &bull; Clip: last {args.clip_duration:g}s" if args.clip_duration > 0 else ""
    if args.detector == "yamnet":
        gate_info = ""
        if args.yamnet_context_rms_ratio > 0 and args.yamnet_min_window_rms > 0:
            gate_info = (
                f" &bull; Energy gate: max({args.yamnet_min_window_rms:.4f}, "
                f"{args.yamnet_context_rms_ratio:.2f}x recent)"
            )
        elif args.yamnet_context_rms_ratio > 0:
            gate_info = f" &bull; Energy gate: {args.yamnet_context_rms_ratio:.2f}x recent"
        elif args.yamnet_min_window_rms > 0:
            gate_info = f" &bull; Energy floor: {args.yamnet_min_window_rms:.4f} RMS"
        return (
            f"{file_count} file(s) &bull; Detector: YAMNet &bull; "
            f"Sample rate: {YAMNET_SAMPLE_RATE} Hz &bull; "
            f"Vocalization threshold: {args.yamnet_score_threshold:.2f}"
            f"{gate_info}{clip_info}"
        )
    return (
        f"{file_count} file(s) &bull; "
        f"Female: {args.female_freq[0]}–{args.female_freq[1]} Hz &bull; "
        f"Male: {args.male_freq[0]}–{args.male_freq[1]} Hz &bull; "
        f"Threshold: {args.threshold}× &bull; "
        f"Noise floor: {args.noise_floor_pct}th pct &bull; "
        f"Min sustained: {args.min_vocal_duration}s{clip_info}"
    )

# ---------------------------------------------------------------------------
# Embedded CSS
# ---------------------------------------------------------------------------

CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #0d0d14;
  --surface:  #1a1a26;
  --surface2: #252535;
  --border:   #2e2e42;
  --text:     #e2e2ee;
  --dim:      #7878a0;
  --pinkf:    #e91e8c;
  --bluem:    #2196f3;
  --gold:     #ffc107;
  --green:    #4caf50;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  padding: 24px;
  min-height: 100vh;
}

h1 { font-size: 26px; font-weight: 700; margin-bottom: 4px; letter-spacing: -0.5px; }
h1 span { color: var(--pinkf); }

.subtitle {
  color: var(--dim);
  font-size: 13px;
  margin-bottom: 22px;
}

/* ── Controls ── */
.controls {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 20px 18px;
  margin-bottom: 20px;
}

.controls-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--dim);
  margin-bottom: 14px;
}

.sliders {
  display: flex;
  flex-wrap: wrap;
  gap: 16px 28px;
  align-items: center;
}

.slider-group {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
  min-width: 220px;
}

.slider-group label {
  font-size: 13px;
  white-space: nowrap;
  min-width: 90px;
}

.slider-group label .swatch {
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 2px;
  margin-right: 5px;
  vertical-align: middle;
}

.slider-group input[type=range] {
  flex: 1;
  accent-color: var(--pinkf);
  cursor: pointer;
}

.slider-val {
  min-width: 28px;
  text-align: right;
  color: var(--dim);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

/* ── Table ── */
.table-wrap { overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }

table { width: 100%; border-collapse: collapse; }

th {
  background: var(--surface);
  padding: 10px 14px;
  text-align: left;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: var(--dim);
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid var(--border);
  transition: color 0.15s;
}

th:hover { color: var(--text); }
th.sort-asc::after  { content: " ↑"; color: var(--pinkf); }
th.sort-desc::after { content: " ↓"; color: var(--pinkf); }

td {
  padding: 8px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}

tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: rgba(255,255,255,0.018); }
tr:hover td { background: var(--surface2); }

.rank-cell { color: var(--dim); font-size: 13px; text-align: right; padding-right: 8px; width: 36px; }

.name-link {
  display: flex;
  align-items: center;
  gap: 9px;
  color: var(--text);
  text-decoration: none;
  transition: color 0.15s;
}
.name-link:hover { color: var(--pinkf); }

.play-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px; height: 24px;
  background: var(--pinkf);
  border-radius: 50%;
  font-size: 10px;
  flex-shrink: 0;
  opacity: 0.75;
  transition: opacity 0.15s;
}
.name-link:hover .play-icon { opacity: 1; }

.filename-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 380px;
}

.dur-cell { font-variant-numeric: tabular-nums; color: var(--dim); }

/* Percentage bar cells */
.bar-cell { position: relative; min-width: 90px; }
.bar-fill {
  position: absolute;
  inset: 0;
  opacity: 0.18;
  border-radius: 0;
  pointer-events: none;
  transition: width 0.3s;
}
.bar-text { position: relative; font-variant-numeric: tabular-nums; }
.bar-f .bar-fill { background: var(--pinkf); }
.bar-m .bar-fill { background: var(--bluem); }
.bar-s .bar-fill { background: var(--gold); }

/* ── Modal overlay ── */
.modal {
  position: fixed;
  inset: 0;
  z-index: 900;
  display: flex;
  align-items: center;
  justify-content: center;
}
.modal.hidden { display: none; }

.modal-backdrop {
  position: absolute;
  inset: 0;
  background: rgba(0,0,0,0.92);
  cursor: pointer;
}

/* ── Player ── */
.player-wrap {
  position: relative;
  z-index: 1;
  width: min(92vw, 1440px);
  background: #000;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 28px 90px rgba(0,0,0,0.85);
  display: flex;
  flex-direction: column;
}

.player-header {
  display: none;
}

.player-title-wrap {
  flex: 1;
  min-width: 0;
}

.player-title {
  font-size: 13px;
  color: #bbb;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.player-subtitle {
  margin-top: 3px;
  font-size: 11px;
  color: #777;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.hdr-btn {
  background: none;
  border: none;
  color: #888;
  font-size: 18px;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 5px;
  line-height: 1;
  transition: background 0.15s, color 0.15s;
  flex-shrink: 0;
}
.hdr-btn:hover { background: rgba(255,255,255,0.1); color: #fff; }

.player-close-overlay {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 12;
  background: rgba(0,0,0,0.62);
  backdrop-filter: blur(8px);
}

/* Video / audio viewport only — keeps the cover from overlapping the control bar */
.player-media {
  position: relative;
  flex-shrink: 0;
  background: #000;
}

#player {
  width: 100%;
  max-height: 80vh;
  display: block;
  background: #000;
  cursor: pointer;
}

/* Audio placeholder (shown when an audio file is loaded into the video element) */
.audio-cover {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 18px;
  background: linear-gradient(160deg, #1a1a26 0%, #0d0d14 100%);
  color: var(--dim);
  text-align: center;
  padding: 24px;
  pointer-events: none;
  z-index: 5;
}
.audio-cover.hidden { display: none; }
.audio-cover-icon {
  font-size: 96px;
  line-height: 1;
  color: var(--pinkf);
  opacity: 0.55;
  text-shadow: 0 4px 24px rgba(233, 30, 140, 0.35);
}
.audio-cover-name {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
  color: #bbb;
  max-width: 80%;
  word-break: break-all;
}

/* When an audio file is loaded the <video> element has no intrinsic size;
   give the media area a sensible minimum height so the cover has room to render. */
.player-wrap.is-audio .player-media {
  min-height: 320px;
}
.player-wrap.is-audio #player {
  min-height: 320px;
  height: 320px;
}

/* Format error overlay */
.player-error {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: rgba(0,0,0,0.88);
  color: #ccc;
  text-align: center;
  padding: 24px;
  z-index: 10;
}
.player-error.hidden { display: none; }
.player-error .err-icon { font-size: 40px; }
.player-error .err-title { font-size: 16px; font-weight: 600; color: #eee; }
.player-error .err-detail { font-size: 12px; color: #888; font-family: monospace; }
.player-error .err-path {
  margin-top: 4px;
  padding: 8px 12px;
  background: rgba(255,255,255,0.06);
  border: 1px solid #333;
  border-radius: 5px;
  font-family: monospace;
  font-size: 12px;
  color: #aaa;
  word-break: break-all;
  max-width: 560px;
  cursor: text;
  user-select: all;
}
.player-error .err-copy {
  padding: 7px 16px;
  background: var(--pinkf);
  color: #fff;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.15s;
}
.player-error .err-copy:hover { opacity: 0.82; }
.player-error .err-note { font-size: 11px; color: #666; max-width: 420px; }

/* Player controls */
.player-controls {
  background: linear-gradient(transparent, rgba(0,0,0,0.88));
  padding: 10px 14px 14px;
  flex-shrink: 0;
  transition: opacity 0.3s;
}

.player-controls.hidden-ctrl {
  opacity: 0;
  pointer-events: none;
}

.progress-area {
  display: flex;
  align-items: center;
  gap: 9px;
  margin-bottom: 9px;
}

.time-txt {
  font-size: 12px;
  color: #aaa;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
  min-width: 42px;
}

.progress-track {
  flex: 1;
  position: relative;
  height: 5px;
  background: rgba(255,255,255,0.18);
  border-radius: 3px;
  cursor: pointer;
  transition: height 0.15s;
}
.progress-track:hover { height: 7px; }

.hotspot-track {
  position: absolute;
  inset: -4px 0;
  pointer-events: none;
  z-index: 3;
}

.hotspot-marker {
  position: absolute;
  top: 0;
  height: 100%;
  min-width: 8px;
  border: none;
  border-radius: 999px;
  background: rgba(255, 193, 7, 0.38);
  box-shadow: 0 0 0 1px rgba(255, 193, 7, 0.2);
  pointer-events: auto;
  cursor: pointer;
  transition: background 0.15s, box-shadow 0.15s, transform 0.15s;
}

.hotspot-marker:hover,
.hotspot-marker.active {
  background: rgba(255, 193, 7, 0.85);
  box-shadow: 0 0 0 1px rgba(255, 193, 7, 0.75);
  transform: scaleY(1.1);
}

.progress-buffered {
  position: absolute;
  top: 0; left: 0; height: 100%;
  background: rgba(255,255,255,0.25);
  border-radius: 3px;
  pointer-events: none;
  z-index: 1;
}

.progress-fill {
  position: absolute;
  top: 0; left: 0; height: 100%;
  background: var(--pinkf);
  border-radius: 3px;
  pointer-events: none;
  z-index: 2;
}

.progress-thumb {
  position: absolute;
  top: 50%;
  width: 14px; height: 14px;
  background: var(--pinkf);
  border-radius: 50%;
  transform: translate(-50%, -50%) scale(0);
  pointer-events: none;
  transition: transform 0.15s;
  z-index: 4;
}
.progress-track:hover .progress-thumb { transform: translate(-50%, -50%) scale(1); }

.hotspot-panel {
  margin-bottom: 10px;
  padding: 8px 10px 9px;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  background: rgba(255,255,255,0.04);
}

.hotspot-panel.hidden { display: none; }

.hotspot-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.hotspot-summary {
  font-size: 12px;
  color: #bcbcd6;
}

.file-nav-row {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 10px;
}

.file-nav-row .player-title-wrap {
  margin-right: auto;
}

.hotspot-settings {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 8px;
}

.hotspot-setting {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.hotspot-setting label {
  font-size: 11px;
  color: #bcbcd6;
}

.hotspot-setting input {
  width: 100%;
  padding: 5px 7px;
  border-radius: 6px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(0,0,0,0.18);
  color: #fff;
  font-size: 12px;
}

.hotspot-setting input:focus {
  outline: none;
  border-color: rgba(255, 193, 7, 0.55);
  box-shadow: 0 0 0 2px rgba(255, 193, 7, 0.12);
}

.hotspot-settings-note {
  margin-bottom: 8px;
  font-size: 11px;
  color: rgba(255,255,255,0.58);
}

.hotspot-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  max-height: 82px;
  overflow-y: auto;
}

.hotspot-chip {
  border: 1px solid rgba(255, 193, 7, 0.22);
  background: rgba(255, 193, 7, 0.08);
  color: #f7e4a0;
  border-radius: 999px;
  padding: 5px 9px;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
}

.hotspot-chip:hover,
.hotspot-chip.active {
  background: rgba(255, 193, 7, 0.2);
  border-color: rgba(255, 193, 7, 0.55);
  color: #fff0b3;
}

.ctrl-row {
  display: flex;
  align-items: center;
  gap: 6px;
}

.ctrl-btn {
  background: none;
  border: none;
  color: #ccc;
  cursor: pointer;
  padding: 6px 8px;
  border-radius: 5px;
  font-size: 16px;
  line-height: 1;
  transition: color 0.15s, background 0.15s;
  flex-shrink: 0;
}
.ctrl-btn:hover { background: rgba(255,255,255,0.1); color: #fff; }
.ctrl-btn:disabled {
  opacity: 0.38;
  cursor: default;
}
.ctrl-btn.active {
  background: var(--pinkf);
  color: #fff;
  font-weight: 700;
}

.ctrl-btn.hidden {
  display: none;
}

.vol-slider {
  width: 80px;
  accent-color: var(--pinkf);
  cursor: pointer;
}

.spacer { flex: 1; }

.kbd-hint {
  font-size: 11px;
  color: #555;
  padding-left: 6px;
  white-space: nowrap;
}

@media (max-width: 900px) {
  .hotspot-settings {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 560px) {
  .hotspot-settings {
    grid-template-columns: 1fr;
  }
}

.hidden-metric { display: none !important; }

/* Cursor hiding when controls hidden and playing */
.player-wrap.hide-cursor { cursor: none; }
"""

# ---------------------------------------------------------------------------
# Embedded JavaScript
# ---------------------------------------------------------------------------

JS = r"""
// ── Data injected above ───────────────────────────────────────────────────

const maxDuration = Math.max(...DATA.map(d => d.duration), 1);
const DEFAULT_HOTSPOT_CONFIG = {
  mergeGap: <<<HOTSPOT_MERGE_GAP>>>,
  padding: <<<HOTSPOT_PADDING>>>,
  minDuration: <<<HOTSPOT_MIN_DURATION>>>,
  seekPreroll: <<<HOTSPOT_SEEK_PREROLL>>>,
};
let hotspotConfig = { ...DEFAULT_HOTSPOT_CONFIG };
let hotspotRefreshTimer = null;

DATA.forEach((item) => {
  item.segments = Array.isArray(item.segments) ? item.segments : [];
  item._serverHotspots = Array.isArray(item.hotspots) ? item.hotspots.slice() : [];
});
const maxPositiveDuration = Math.max(...DATA.map(d => d.positive_duration || 0), 1);

// ── Scoring ───────────────────────────────────────────────────────────────

function getWeights() {
  return {
    dur: parseFloat(document.getElementById('w-dur').value),
    fem: parseFloat(document.getElementById('w-fem').value),
    mal: parseFloat(document.getElementById('w-mal').value),
    eng: parseFloat(document.getElementById('w-eng').value),
  };
}

function computeScore(item, w) {
  const energy = Math.max(0, Math.min(1, item.energy_confidence || 0));
  const positiveTime = (item.positive_duration || 0) / maxPositiveDuration;
  const base =
    w.dur * positiveTime +
    w.fem * (item.female_pct / 100) +
    (HAS_SECOND_METRIC ? w.mal * (item.male_pct / 100) : 0);
  const damping = 0.35 + 0.65 * energy;
  return base * damping + w.eng * energy;
}

// ── Table rendering ───────────────────────────────────────────────────────

let sortCol = 'score';
let sortAsc  = false;
let playlistOrder = DATA.map((_, idx) => idx);

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderTable() {
  const w = getWeights();
  const rows = DATA.map((item, idx) => ({
    ...item, _idx: idx, score: computeScore(item, w),
  }));

  rows.sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ?  1 : -1;
    return 0;
  });

  const maxScore = Math.max(...rows.map(r => r.score), 0.001);

  const tbody = document.getElementById('tbody');
  playlistOrder = rows.map(item => item._idx);
  tbody.innerHTML = rows.map((item, rank) => {
    const scoreNorm = (item.score / maxScore) * 100;
    const femW = Math.min(item.female_pct, 100);
    const malW = Math.min(item.male_pct,  100);
    const femTxt = item.female_pct >= 0 ? item.female_pct.toFixed(1) + '%' : '—';
    const malTxt = item.male_pct   >= 0 ? item.male_pct.toFixed(1)   + '%' : '—';
    const metric2Cell = HAS_SECOND_METRIC ? `
      <td class="bar-cell bar-m">
        <div class="bar-fill" style="width:${malW}%"></div>
        <span class="bar-text">${malTxt}</span>
      </td>` : '';

    const icon = item.kind === 'audio' ? '\u266A' : '\u25B6';
    return `<tr>
      <td class="rank-cell">${rank + 1}</td>
      <td>
        <a class="name-link" href="#"
           data-idx="${item._idx}"
           onclick="openPlayer(${item._idx});return false;">
          <span class="play-icon">${icon}</span>
          <span class="filename-text" title="${esc(item.name)}">${esc(item.name)}</span>
        </a>
      </td>
      <td class="dur-cell">${item.duration_fmt}</td>
      <td class="bar-cell bar-f">
        <div class="bar-fill" style="width:${femW}%"></div>
        <span class="bar-text">${femTxt}</span>
      </td>
      ${metric2Cell}
      <td class="bar-cell bar-s">
        <div class="bar-fill" style="width:${scoreNorm.toFixed(1)}%"></div>
        <span class="bar-text">${(item.score * 100).toFixed(1)}</span>
      </td>
    </tr>`;
  }).join('');

  // Update sort indicators
  document.querySelectorAll('th[data-col]').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.col === sortCol)
      th.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');
  });

  if (currentItemIdx !== null) {
    syncFileButtons();
  }
}

// Sliders
['w-dur','w-fem','w-mal','w-eng'].forEach(id => {
  const sl = document.getElementById(id);
  if (!sl) return;
  const vl = document.getElementById(id + '-val');
  sl.addEventListener('input', () => {
    vl.textContent = parseFloat(sl.value).toFixed(1);
    renderTable();
  });
});

// Column sort
document.querySelectorAll('th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (sortCol === col) { sortAsc = !sortAsc; }
    else { sortCol = col; sortAsc = (col === 'name'); }
    renderTable();
  });
});

// ── Video Player ──────────────────────────────────────────────────────────

const modal          = document.getElementById('player-modal');
const playerWrap     = document.getElementById('player-wrap');
const video          = document.getElementById('player');
const playerControls = document.getElementById('player-controls');
const playerTitleEl   = document.getElementById('player-title');
const playerSubtitleEl = document.getElementById('player-subtitle');
const playerError     = document.getElementById('player-error');
const playerErrDetail = document.getElementById('player-error-detail');
const playerErrPath   = document.getElementById('player-error-path');
const audioCover      = document.getElementById('audio-cover');
const audioCoverName  = document.getElementById('audio-cover-name');
const playerMedia     = document.querySelector('.player-media');
const progressTrack  = document.getElementById('progress-track');
const hotspotTrack   = document.getElementById('hotspot-track');
const progressFill   = document.getElementById('progress-fill');
const progressBuf    = document.getElementById('progress-buf');
const progressThumb  = document.getElementById('progress-thumb');
const timeCurEl      = document.getElementById('time-cur');
const timeTotEl      = document.getElementById('time-tot');
const playBtn        = document.getElementById('btn-play');
const autoNextBtn    = document.getElementById('btn-auto-next');
const airplayBtn     = document.getElementById('btn-airplay');
const muteBtn        = document.getElementById('btn-mute');
const volSlider      = document.getElementById('vol-track');
const hotspotPanel   = document.getElementById('hotspot-panel');
const hotspotSummary = document.getElementById('hotspot-summary');
const hotspotList    = document.getElementById('hotspot-list');
const prevHotspotBtn = document.getElementById('btn-prev-hotspot');
const nextHotspotBtn = document.getElementById('btn-next-hotspot');
const prevFileBtn    = document.getElementById('btn-prev-file');
const nextFileBtn    = document.getElementById('btn-next-file');
const hotspotMergeGapInput = document.getElementById('hotspot-merge-gap');
const hotspotPaddingInput = document.getElementById('hotspot-padding');
const hotspotMinDurationInput = document.getElementById('hotspot-min-duration');
const hotspotSeekPrerollInput = document.getElementById('hotspot-seek-preroll');

let isDragging    = false;
let savedVol      = 1;
let hideCtrlTimer = null;
let currentItemIdx = null;
let autoNext       = false;
let activeHotspots = [];
let activeHotspotIdx = -1;
let autoAdvancePendingHotspotIdx = -1;
let pendingSeekTime = null;
let airplayAvailable = false;

function clamp01(n) {
  return Math.max(0, Math.min(1, n));
}

function roundTo(value, digits) {
  return Number(value.toFixed(digits));
}

function segmentStrength(segment) {
  for (const key of ['peak_score', 'avg_score', 'peak_ratio', 'avg_ratio']) {
    const value = Number(segment?.[key]);
    if (Number.isFinite(value)) return value;
  }
  return 0;
}

function unionDuration(intervals) {
  if (!intervals.length) return 0;
  let total = 0;
  let [curStart, curEnd] = intervals[0];
  for (const [start, end] of intervals.slice(1)) {
    if (start <= curEnd) {
      curEnd = Math.max(curEnd, end);
      continue;
    }
    total += Math.max(0, curEnd - curStart);
    [curStart, curEnd] = [start, end];
  }
  total += Math.max(0, curEnd - curStart);
  return total;
}

function buildHotspotsFromSegments(segments, duration, config) {
  if (!Array.isArray(segments) || segments.length === 0) return [];
  const maxDur = Number.isFinite(duration) && duration > 0 ? duration : null;
  const ordered = segments
    .filter((seg) => Number(seg?.end) > Number(seg?.start))
    .slice()
    .sort((a, b) => Number(a.start) - Number(b.start) || Number(a.end) - Number(b.end));
  if (!ordered.length) return [];

  const clusters = [];
  let current = [ordered[0]];
  let currentEnd = Number(ordered[0].end);
  for (const seg of ordered.slice(1)) {
    const segStart = Number(seg.start);
    const segEnd = Number(seg.end);
    if (segStart <= currentEnd + config.mergeGap) {
      current.push(seg);
      currentEnd = Math.max(currentEnd, segEnd);
      continue;
    }
    clusters.push(current);
    current = [seg];
    currentEnd = segEnd;
  }
  clusters.push(current);

  return clusters.map((cluster) => {
    const rawStart = Number(cluster[0].start);
    const rawEnd = Math.max(...cluster.map((seg) => Number(seg.end)));
    const rawSpan = Math.max(0, rawEnd - rawStart);
    if (rawSpan <= 0) return null;

    const intervals = cluster
      .map((seg) => [Number(seg.start), Number(seg.end)])
      .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const positiveDuration = unionDuration(intervals);
    const density = rawSpan > 0 ? positiveDuration / rawSpan : 0;
    const peakStrength = Math.max(...cluster.map((seg) => segmentStrength(seg)), 0);
    const labels = [...new Set(cluster.map((seg) => String(seg.label || 'vocalization')))].sort();

    let navStart = Math.max(0, rawStart - config.padding);
    let navEnd = rawEnd + config.padding;
    if (maxDur !== null) navEnd = Math.min(maxDur, navEnd);
    let navSpan = navEnd - navStart;

    if (navSpan < config.minDuration) {
      const center = (rawStart + rawEnd) / 2;
      const half = config.minDuration / 2;
      navStart = Math.max(0, center - half);
      navEnd = center + half;
      if (maxDur !== null) {
        if (navEnd > maxDur) {
          navEnd = maxDur;
          navStart = Math.max(0, navEnd - config.minDuration);
        } else {
          navStart = Math.max(0, navStart);
        }
      }
      navSpan = navEnd - navStart;
    }

    let seekTo = Math.max(0, rawStart - config.seekPreroll);
    if (maxDur !== null) seekTo = Math.min(maxDur, seekTo);

    return {
      start: roundTo(navStart, 3),
      end: roundTo(navEnd, 3),
      seek_to: roundTo(seekTo, 3),
      duration: roundTo(Math.max(0, navSpan), 3),
      raw_start: roundTo(rawStart, 3),
      raw_end: roundTo(rawEnd, 3),
      positive_duration: roundTo(positiveDuration, 3),
      density: roundTo(density, 4),
      peak_strength: roundTo(peakStrength, 4),
      segment_count: cluster.length,
      labels,
    };
  }).filter(Boolean);
}

function parseHotspotConfigValue(input, fallback, min = 0, max = 999) {
  const value = Number(input?.value);
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function readHotspotConfigFromInputs() {
  hotspotConfig = {
    mergeGap: parseHotspotConfigValue(hotspotMergeGapInput, DEFAULT_HOTSPOT_CONFIG.mergeGap, 0, 240),
    padding: parseHotspotConfigValue(hotspotPaddingInput, DEFAULT_HOTSPOT_CONFIG.padding, 0, 240),
    minDuration: parseHotspotConfigValue(hotspotMinDurationInput, DEFAULT_HOTSPOT_CONFIG.minDuration, 1, 360),
    seekPreroll: parseHotspotConfigValue(hotspotSeekPrerollInput, DEFAULT_HOTSPOT_CONFIG.seekPreroll, 0, 60),
  };
}

function writeHotspotConfigToInputs() {
  hotspotMergeGapInput.value = hotspotConfig.mergeGap;
  hotspotPaddingInput.value = hotspotConfig.padding;
  hotspotMinDurationInput.value = hotspotConfig.minDuration;
  hotspotSeekPrerollInput.value = hotspotConfig.seekPreroll;
}

function canUseAirPlay() {
  return typeof video.webkitShowPlaybackTargetPicker === 'function';
}

function updateAirPlayButton() {
  const supported = canUseAirPlay();
  const wireless = !!video.webkitCurrentPlaybackTargetIsWireless;
  const visible = supported;
  airplayBtn.classList.toggle('hidden', !visible);
  airplayBtn.classList.toggle('active', wireless);
  airplayBtn.disabled = !supported;
  airplayBtn.setAttribute('aria-pressed', wireless ? 'true' : 'false');
  airplayBtn.textContent = wireless ? 'AirPlay: On' : 'AirPlay';
  airplayBtn.title = !supported
    ? 'AirPlay is only available in Safari/WebKit browsers that support external playback'
    : wireless
      ? 'AirPlay playback target is active'
      : airplayAvailable
        ? 'Choose an AirPlay playback target'
        : 'Open the AirPlay playback target picker';
}

function openAirPlayPicker() {
  if (!canUseAirPlay()) return;
  try {
    video.webkitShowPlaybackTargetPicker();
  } catch (_err) {
    return;
  }
  showControls();
}

function setHotspotInputsDisabled(disabled) {
  [hotspotMergeGapInput, hotspotPaddingInput, hotspotMinDurationInput, hotspotSeekPrerollInput].forEach((input) => {
    input.disabled = disabled;
  });
}

function itemHasAdjustableHotspots(item) {
  return !!(item && Array.isArray(item.segments) && item.segments.length > 0);
}

function deriveHotspotsForItem(item) {
  if (itemHasAdjustableHotspots(item)) {
    return buildHotspotsFromSegments(item.segments, item.duration, hotspotConfig);
  }
  return Array.isArray(item?._serverHotspots) ? item._serverHotspots.slice() : [];
}

function refreshHotspotsForAllItems() {
  DATA.forEach((item) => {
    item.hotspots = deriveHotspotsForItem(item);
  });
}

function refreshCurrentItemHotspots() {
  if (currentItemIdx === null) return;
  const item = DATA[currentItemIdx];
  item.hotspots = deriveHotspotsForItem(item);
  renderHotspots(item.hotspots || []);
  updateProgress();
}

function scheduleHotspotRefresh() {
  clearTimeout(hotspotRefreshTimer);
  hotspotRefreshTimer = setTimeout(() => {
    readHotspotConfigFromInputs();
    refreshHotspotsForAllItems();
    refreshCurrentItemHotspots();
  }, 100);
}

function hotspotTitle(hotspot, idx) {
  return `Hotspot ${idx + 1}: ${fmtTime(hotspot.raw_start ?? hotspot.start)}-${fmtTime(hotspot.raw_end ?? hotspot.end)} · ${(hotspot.density * 100).toFixed(0)}% dense`;
}

function hotspotChipText(hotspot, idx) {
  const start = fmtTime(hotspot.raw_start ?? hotspot.start);
  const density = `${Math.round((hotspot.density || 0) * 100)}%`;
  return `#${idx + 1} ${start} · ${density}`;
}

function syncHotspotButtons() {
  const hasHotspots = activeHotspots.length > 0;
  prevHotspotBtn.disabled = !hasHotspots || prevHotspotIndex() === -1;
  nextHotspotBtn.disabled = !hasHotspots || !hasNextHotspotJump();
}

function setActiveHotspot(idx) {
  if (idx !== activeHotspotIdx) {
    autoAdvancePendingHotspotIdx = idx;
  }
  activeHotspotIdx = idx;
  hotspotTrack.querySelectorAll('.hotspot-marker').forEach((el, markerIdx) => {
    el.classList.toggle('active', markerIdx === idx);
  });
  hotspotList.querySelectorAll('.hotspot-chip').forEach((el, chipIdx) => {
    el.classList.toggle('active', chipIdx === idx);
  });
  syncHotspotButtons();
}

function currentHotspotIndexForTime(time) {
  return activeHotspots.findIndex(hotspot => time >= hotspot.start && time <= hotspot.end);
}

function nextItemIndexInPlaylist() {
  if (currentItemIdx === null) return null;
  const pos = playlistOrder.indexOf(currentItemIdx);
  const nextIdx = pos >= 0 ? playlistOrder[pos + 1] : currentItemIdx + 1;
  if (nextIdx === undefined || nextIdx >= DATA.length) return null;
  return nextIdx;
}

function prevItemIndexInPlaylist() {
  if (currentItemIdx === null) return null;
  const pos = playlistOrder.indexOf(currentItemIdx);
  const prevIdx = pos > 0 ? playlistOrder[pos - 1] : currentItemIdx - 1;
  if (prevIdx === undefined || prevIdx < 0) return null;
  return prevIdx;
}

function prevHotspotIndex() {
  if (activeHotspots.length === 0) return -1;
  const time = video.currentTime || 0;
  const containing = currentHotspotIndexForTime(time);
  if (containing >= 0 && time > activeHotspots[containing].seek_to + 1.0) {
    return containing;
  }
  for (let idx = activeHotspots.length - 1; idx >= 0; idx -= 1) {
    if (activeHotspots[idx].seek_to < time - 0.5) return idx;
  }
  return -1;
}

function nextHotspotIndex() {
  if (activeHotspots.length === 0) return -1;
  const time = video.currentTime || 0;
  const containing = currentHotspotIndexForTime(time);
  const minStart = containing >= 0 ? activeHotspots[containing].end + 0.25 : time + 0.5;
  const idx = activeHotspots.findIndex(hotspot => hotspot.seek_to >= minStart);
  return idx;
}

function hasNextHotspotJump() {
  if (nextHotspotIndex() >= 0) return true;
  return nextItemIndexInPlaylist() !== null;
}

function syncFileButtons() {
  prevFileBtn.disabled = prevItemIndexInPlaylist() === null;
  nextFileBtn.disabled = nextItemIndexInPlaylist() === null;
}

function jumpToHotspot(idx) {
  if (idx < 0 || idx >= activeHotspots.length || !video.duration) return;
  const hotspot = activeHotspots[idx];
  video.currentTime = Math.max(0, Math.min(video.duration, hotspot.seek_to));
  setActiveHotspot(idx);
  updateProgress();
  showControls();
}

function firstHotspotSeekForItem(idx) {
  const item = DATA[idx];
  if (!item) return null;
  item.hotspots = deriveHotspotsForItem(item);
  if (!Array.isArray(item.hotspots) || item.hotspots.length === 0) return null;
  return item.hotspots[0].seek_to ?? item.hotspots[0].start ?? null;
}

function jumpPrevHotspot() {
  const idx = prevHotspotIndex();
  if (idx >= 0) jumpToHotspot(idx);
}

function jumpPrevFile() {
  const prevIdx = prevItemIndexInPlaylist();
  if (prevIdx === null) return;
  openPlayer(prevIdx, { seekTime: firstHotspotSeekForItem(prevIdx) ?? 0 });
}

function jumpNextHotspot() {
  const idx = nextHotspotIndex();
  if (idx >= 0) {
    jumpToHotspot(idx);
    return;
  }
  const nextIdx = nextItemIndexInPlaylist();
  if (nextIdx === null) return;
  openPlayer(nextIdx, { seekTime: firstHotspotSeekForItem(nextIdx) ?? 0 });
}

function renderHotspots(hotspots) {
  const item = currentItemIdx !== null ? DATA[currentItemIdx] : null;
  const hasAdjustableHotspots = itemHasAdjustableHotspots(item);
  setHotspotInputsDisabled(!hasAdjustableHotspots);
  activeHotspots = Array.isArray(hotspots) ? hotspots.slice().sort((a, b) => a.start - b.start) : [];
  activeHotspotIdx = -1;
  autoAdvancePendingHotspotIdx = -1;
  hotspotTrack.innerHTML = '';
  hotspotList.innerHTML = '';

  if (activeHotspots.length === 0) {
    if (!hasAdjustableHotspots) {
      hotspotPanel.classList.add('hidden');
      hotspotSummary.textContent = '';
      syncHotspotButtons();
      return;
    }
    hotspotPanel.classList.remove('hidden');
    hotspotSummary.textContent = 'No hotspots at the current clustering settings.';
    syncHotspotButtons();
    return;
  }

  hotspotPanel.classList.remove('hidden');
  hotspotSummary.textContent = `${activeHotspots.length} hotspot${activeHotspots.length === 1 ? '' : 's'} from dense positive regions`;

  hotspotTrack.innerHTML = activeHotspots.map((hotspot, idx) => {
    const left = clamp01((hotspot.start || 0) / (video.duration || hotspot.end || 1)) * 100;
    const width = Math.max(
      0.9,
      (clamp01((hotspot.end || 0) / (video.duration || hotspot.end || 1)) * 100) - left,
    );
    return `<button type="button" class="hotspot-marker" data-hotspot-idx="${idx}" style="left:${left.toFixed(3)}%;width:${width.toFixed(3)}%" title="${esc(hotspotTitle(hotspot, idx))}"></button>`;
  }).join('');

  hotspotList.innerHTML = activeHotspots.map((hotspot, idx) =>
    `<button type="button" class="hotspot-chip" data-hotspot-idx="${idx}" title="${esc(hotspotTitle(hotspot, idx))}">${esc(hotspotChipText(hotspot, idx))}</button>`
  ).join('');

  hotspotTrack.querySelectorAll('[data-hotspot-idx]').forEach((el) => {
    el.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      e.preventDefault();
    });
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = parseInt(el.dataset.hotspotIdx, 10);
      if (!Number.isNaN(idx)) jumpToHotspot(idx);
    });
  });
  hotspotList.querySelectorAll('[data-hotspot-idx]').forEach((el) => {
    el.addEventListener('click', () => {
      const idx = parseInt(el.dataset.hotspotIdx, 10);
      if (!Number.isNaN(idx)) jumpToHotspot(idx);
    });
  });

  setActiveHotspot(currentHotspotIndexForTime(video.currentTime || 0));
}

// Open / close
function mediaSrcForItem(item, idx) {
  // Safari blocks file:// media outside the report's folder. When the report
  // is served over localhost, stream via /media/<idx> instead.
  if (location.protocol === 'http:' || location.protocol === 'https:') {
    return '/media/' + idx;
  }
  return item.url;
}

function openPlayer(idx, opts = {}) {
  const item = DATA[idx];
  const isAudio = item.kind === 'audio';
  currentItemIdx = idx;
  const defaultSeekTime = firstHotspotSeekForItem(idx);
  const requestedSeekTime = Number.isFinite(opts.seekTime) ? opts.seekTime : defaultSeekTime;
  pendingSeekTime = Number.isFinite(requestedSeekTime) ? requestedSeekTime : null;
  playerTitleEl.textContent = item.name;
  playerSubtitleEl.textContent = item.parent_dir || '';
  playerError.classList.add('hidden');
  playerControls.style.display = '';
  updateAutoNextButton();
  playerWrap.classList.toggle('is-audio', isAudio);
  audioCover.classList.toggle('hidden', !isAudio);
  audioCoverName.textContent = isAudio ? item.name : '';
  item.hotspots = deriveHotspotsForItem(item);
  airplayAvailable = false;
  updateAirPlayButton();
  video.src = mediaSrcForItem(item, idx);
  video.load();
  renderHotspots(item.hotspots || []);
  syncFileButtons();
  modal.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
  showControls();
  video.play().catch(() => {});
}

video.addEventListener('error', () => {
  const e = video.error;
  const CODE = {1:'ABORTED', 2:'NETWORK', 3:'DECODE', 4:'SRC_NOT_SUPPORTED'};
  const detail = e ? `${CODE[e.code] || 'ERR'} (code ${e.code})${e.message ? ': ' + e.message : ''}` : 'Unknown error';
  const item = currentItemIdx != null ? DATA[currentItemIdx] : null;
  // Prefer the real filesystem path; fall back to decoding a file:// URL.
  const rawPath = (item && item.path)
    || decodeURIComponent(video.src.replace(/^file:\/\//, ''));
  playerErrDetail.textContent = detail;
  playerErrPath.textContent = rawPath;
  playerError.classList.remove('hidden');
  playerControls.style.display = 'none';
  clearTimeout(hideCtrlTimer);
});

function copyErrPath() {
  const path = playerErrPath.textContent;
  navigator.clipboard.writeText(path).then(() => {
    const btn = playerError.querySelector('.err-copy');
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 1500);
  });
}

function closePlayer() {
  if (document.fullscreenElement) {
    document.exitFullscreen().then(closePlayer);
    return;
  }
  modal.classList.add('hidden');
  video.pause();
  video.src = '';
  document.body.style.overflow = '';
  playerError.classList.add('hidden');
  playerControls.style.display = '';
  playerWrap.classList.remove('is-audio');
  audioCover.classList.add('hidden');
  audioCoverName.textContent = '';
  playerTitleEl.textContent = '';
  playerSubtitleEl.textContent = '';
  currentItemIdx = null;
  activeHotspots = [];
  activeHotspotIdx = -1;
  autoAdvancePendingHotspotIdx = -1;
  pendingSeekTime = null;
  airplayAvailable = false;
  hotspotTrack.innerHTML = '';
  hotspotList.innerHTML = '';
  hotspotPanel.classList.add('hidden');
  hotspotSummary.textContent = '';
  syncFileButtons();
  updateAirPlayButton();
  updateAutoNextButton();
  clearTimeout(hideCtrlTimer);
}

document.getElementById('modal-backdrop').addEventListener('click', closePlayer);

// Controls visibility
function showControls() {
  playerControls.classList.remove('hidden-ctrl');
  playerWrap.classList.remove('hide-cursor');
  clearTimeout(hideCtrlTimer);
  // Audio has no video frame to "wake" the UI — keep scrubber + volume visible.
  if (playerWrap.classList.contains('is-audio')) return;
  if (!video.paused) {
    hideCtrlTimer = setTimeout(() => {
      playerControls.classList.add('hidden-ctrl');
      playerWrap.classList.add('hide-cursor');
    }, 3000);
  }
}

playerWrap.addEventListener('mousemove', showControls);
playerMedia.addEventListener('mousemove', showControls);
video.addEventListener('play',  () => {
  playBtn.textContent = '⏸';
  if (playerWrap.classList.contains('is-audio')) return;
  hideCtrlTimer = setTimeout(() => {
    playerControls.classList.add('hidden-ctrl');
    playerWrap.classList.add('hide-cursor');
  }, 3000);
});
video.addEventListener('pause', () => {
  playBtn.textContent = '▶';
  clearTimeout(hideCtrlTimer);
  showControls();
});

// Playback controls
function togglePlay() {
  if (video.paused) video.play().catch(() => {});
  else video.pause();
}

function updateAutoNextButton() {
  autoNextBtn.classList.toggle('active', autoNext);
  autoNextBtn.setAttribute('aria-pressed', autoNext ? 'true' : 'false');
  autoNextBtn.textContent = autoNext ? 'Auto next hotspot: On' : 'Auto next hotspot: Off';
  autoNextBtn.title = autoNext
    ? 'Auto-advance through hotspots and into the next file: on'
    : 'Auto-advance through hotspots and into the next file: off';
}

function toggleAutoNext() {
  autoNext = !autoNext;
  updateAutoNextButton();
  showControls();
}

function playNextInSequence() {
  const nextIdx = nextItemIndexInPlaylist();
  if (nextIdx === null) return false;
  const nextSeekTime = autoNext ? firstHotspotSeekForItem(nextIdx) : null;
  openPlayer(nextIdx, { seekTime: nextSeekTime });
  return true;
}

function autoAdvanceFromCurrentHotspot() {
  const idx = activeHotspotIdx >= 0 ? activeHotspotIdx : currentHotspotIndexForTime(video.currentTime || 0);
  if (idx < 0 || idx >= activeHotspots.length) return false;
  const nextIdx = activeHotspots.findIndex((hotspot, hotspotIdx) => hotspotIdx > idx && hotspot.seek_to > activeHotspots[idx].end);
  if (nextIdx >= 0) {
    jumpToHotspot(nextIdx);
    return true;
  }
  return playNextInSequence();
}

function seek(delta) {
  video.currentTime = Math.max(0, Math.min(video.duration || 0, video.currentTime + delta));
}

function toggleMute() {
  if (video.muted) {
    video.muted = false;
    video.volume = savedVol || 1;
  } else {
    savedVol = video.volume;
    video.muted = true;
  }
  updateMuteIcon();
}

function updateMuteIcon() {
  const v = video.volume;
  muteBtn.textContent = (video.muted || v === 0) ? '🔇' : v < 0.4 ? '🔉' : '🔊';
  volSlider.value = video.muted ? 0 : v;
}

volSlider.addEventListener('input', () => {
  video.volume = parseFloat(volSlider.value);
  video.muted  = (video.volume === 0);
  updateMuteIcon();
});

function reqFullscreen() {
  if (!document.fullscreenElement) {
    playerWrap.requestFullscreen().catch(() => video.requestFullscreen());
  } else {
    document.exitFullscreen();
  }
}

// Time formatting
function fmtTime(s) {
  if (isNaN(s) || s < 0) return '0:00';
  s = Math.floor(s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = String(m).padStart(2,'0');
  const ss = String(sec).padStart(2,'0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

// Progress bar
function updateProgress() {
  const cur = video.currentTime;
  const dur = video.duration || 0;
  const pct = dur > 0 ? cur / dur : 0;
  progressFill.style.width  = `${pct * 100}%`;
  progressThumb.style.left  = `${pct * 100}%`;
  timeCurEl.textContent     = fmtTime(cur);
  timeTotEl.textContent     = fmtTime(dur);
  if (video.buffered.length > 0) {
    const bp = video.buffered.end(video.buffered.length - 1) / (dur || 1);
    progressBuf.style.width = `${bp * 100}%`;
  }
  if (activeHotspots.length > 0) {
    const idx = currentHotspotIndexForTime(cur);
    setActiveHotspot(idx);
    if (
      autoNext &&
      idx >= 0 &&
      idx === autoAdvancePendingHotspotIdx &&
      cur >= activeHotspots[idx].end - 0.12
    ) {
      autoAdvancePendingHotspotIdx = -1;
      autoAdvanceFromCurrentHotspot();
      return;
    }
  } else {
    syncHotspotButtons();
  }
}

function seekToFraction(e) {
  const rect = progressTrack.getBoundingClientRect();
  const x = (e.clientX !== undefined ? e.clientX
             : e.touches && e.touches[0] ? e.touches[0].clientX : 0) - rect.left;
  const pct = Math.max(0, Math.min(1, x / rect.width));
  if (video.duration) video.currentTime = pct * video.duration;
  updateProgress();
}

progressTrack.addEventListener('mousedown', e => {
  isDragging = true;
  seekToFraction(e);
});
document.addEventListener('mousemove', e => { if (isDragging) seekToFraction(e); });
document.addEventListener('mouseup',   () => { isDragging = false; });

video.addEventListener('timeupdate',    updateProgress);
video.addEventListener('durationchange', updateProgress);
video.addEventListener('loadedmetadata', () => {
  if (currentItemIdx !== null) renderHotspots(DATA[currentItemIdx].hotspots || []);
  if (pendingSeekTime !== null && Number.isFinite(pendingSeekTime)) {
    const target = Math.max(0, Math.min(video.duration || pendingSeekTime, pendingSeekTime));
    video.currentTime = target;
  }
  pendingSeekTime = null;
  updateAirPlayButton();
  updateProgress();
});
video.addEventListener('webkitplaybacktargetavailabilitychanged', (event) => {
  airplayAvailable = event.availability === 'available';
  updateAirPlayButton();
});
video.addEventListener('webkitcurrentplaybacktargetiswirelesschanged', () => {
  updateAirPlayButton();
});
video.addEventListener('ended',         () => {
  if (!autoNext) return;
  if (activeHotspots.length > 0) {
    autoAdvancePendingHotspotIdx = -1;
  }
  playNextInSequence();
});
video.addEventListener('click',         () => { togglePlay(); showControls(); });
video.addEventListener('dblclick',      reqFullscreen);

// Mouse wheel → volume (video element and whole media area for audio-only playback)
function wheelVolume(e) {
  e.preventDefault();
  video.muted  = false;
  video.volume = Math.max(0, Math.min(1, video.volume + (e.deltaY < 0 ? 0.05 : -0.05)));
  updateMuteIcon();
  showControls();
}
playerMedia.addEventListener('wheel', wheelVolume, { passive: false });

// Keyboard
document.addEventListener('keydown', e => {
  if (modal.classList.contains('hidden')) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  switch (e.key) {
    case ' ': case 'k': case 'K':
      e.preventDefault(); togglePlay(); showControls(); break;
    case 'ArrowLeft':
      e.preventDefault(); seek(e.shiftKey ? -30 : -5); showControls(); break;
    case 'ArrowRight':
      e.preventDefault(); seek(e.shiftKey ?  30 :  5); showControls(); break;
    case '[':
      e.preventDefault(); jumpPrevHotspot(); break;
    case ']':
      e.preventDefault(); jumpNextHotspot(); break;
    case 'ArrowUp':
      e.preventDefault();
      video.volume = Math.min(1, video.volume + 0.1);
      video.muted = false; updateMuteIcon(); showControls(); break;
    case 'ArrowDown':
      e.preventDefault();
      video.volume = Math.max(0, video.volume - 0.1);
      updateMuteIcon(); showControls(); break;
    case 'm': case 'M':
      toggleMute(); showControls(); break;
    case 'f': case 'F':
      reqFullscreen(); break;
    case 'Escape':
      if (document.fullscreenElement) document.exitFullscreen();
      else closePlayer();
      break;
    default:
      if (e.key >= '0' && e.key <= '9' && video.duration) {
        video.currentTime = video.duration * (parseInt(e.key) / 10);
        showControls();
      }
  }
});

// Init
writeHotspotConfigToInputs();
setHotspotInputsDisabled(true);
[hotspotMergeGapInput, hotspotPaddingInput, hotspotMinDurationInput, hotspotSeekPrerollInput].forEach((input) => {
  input.addEventListener('input', scheduleHotspotRefresh);
  input.addEventListener('change', scheduleHotspotRefresh);
});
refreshHotspotsForAllItems();
updateAutoNextButton();
renderTable();
"""

# ---------------------------------------------------------------------------
# HTML template  (uses <<<MARKERS>>> to avoid f-string brace conflicts)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ScreamFinder — <<<FILE_COUNT>>> file(s)</title>
  <style><<<CSS>>></style>
</head>
<body>

<h1><span>Scream</span>Finder</h1>
<p class="subtitle">
  <<<SUBTITLE>>>
</p>

<div class="controls">
  <div class="controls-title">Sort Weights</div>
  <div class="sliders">
    <div class="slider-group">
      <label><span class="swatch" style="background:#7878a0"></span>Positive Time</label>
      <input type="range" id="w-dur" min="0" max="5" step="0.1" value="1.0">
      <span class="slider-val" id="w-dur-val">1.0</span>
    </div>
    <div class="slider-group">
      <label><span class="swatch" style="background:#e91e8c"></span><<<METRIC1_LABEL>>></label>
      <input type="range" id="w-fem" min="0" max="5" step="0.1" value="2.0">
      <span class="slider-val" id="w-fem-val">2.0</span>
    </div>
    <div class="slider-group <<<METRIC2_GROUP_CLASS>>>">
      <label><span class="swatch" style="background:#2196f3"></span><<<METRIC2_LABEL>>></label>
      <input type="range" id="w-mal" min="0" max="5" step="0.1" value="1.0">
      <span class="slider-val" id="w-mal-val">1.0</span>
    </div>
    <div class="slider-group">
      <label><span class="swatch" style="background:#5ec9a1"></span>Energy</label>
      <input type="range" id="w-eng" min="0" max="5" step="0.1" value="2.0">
      <span class="slider-val" id="w-eng-val">2.0</span>
    </div>
  </div>
</div>

<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th style="width:36px">#</th>
        <th data-col="name">Filename</th>
        <th data-col="duration">Duration</th>
        <th data-col="female_pct"><<<METRIC1_LABEL>>></th>
        <th data-col="male_pct" class="<<<METRIC2_COL_CLASS>>>"><<<METRIC2_LABEL>>></th>
        <th data-col="score">Score</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<!-- Video Player Modal -->
  <div id="player-modal" class="modal hidden">
  <div id="modal-backdrop" class="modal-backdrop"></div>
  <div id="player-wrap" class="player-wrap">
    <div class="player-header">
      <button class="hdr-btn" onclick="closePlayer()" title="Close (Esc)">✕</button>
    </div>
    <div class="player-media">
      <button class="hdr-btn player-close-overlay" onclick="closePlayer()" title="Close (Esc)">✕</button>
      <video id="player" preload="metadata" playsinline webkit-playsinline x-webkit-airplay="allow" airplay="allow"></video>
      <div id="audio-cover" class="audio-cover hidden">
        <div class="audio-cover-icon">&#9835;</div>
        <div id="audio-cover-name" class="audio-cover-name"></div>
      </div>
      <div id="player-error" class="player-error hidden">
      <div class="err-icon">⚠</div>
      <div class="err-title">This file cannot be played in the browser</div>
      <div id="player-error-detail" class="err-detail"></div>
      <div id="player-error-path" class="err-path" title="Click to select all"></div>
      <button class="err-copy" onclick="copyErrPath()">Copy path</button>
      <div class="err-note">
        Paste this path into VLC, IINA, QuickTime, or any other media player.<br>
        Browsers only support MP4/H.264, WebM, and Ogg natively.<br>
        Safari blocks <code>file://</code> media outside the report folder — use
        <code>python3 screamfinder.py --serve-report &lt;this-file.html&gt;</code>.
      </div>
    </div>
    </div>
    <div id="player-controls" class="player-controls">
      <div class="progress-area">
        <span id="time-cur" class="time-txt">0:00</span>
        <div id="progress-track" class="progress-track">
          <div id="hotspot-track" class="hotspot-track"></div>
          <div id="progress-buf"   class="progress-buffered" style="width:0%"></div>
          <div id="progress-fill"  class="progress-fill"     style="width:0%"></div>
          <div id="progress-thumb" class="progress-thumb"    style="left:0%"></div>
        </div>
        <span id="time-tot" class="time-txt" style="text-align:right">0:00</span>
      </div>
      <div id="hotspot-panel" class="hotspot-panel hidden">
        <div class="hotspot-toolbar">
          <span id="hotspot-summary" class="hotspot-summary"></span>
          <span class="spacer"></span>
          <button id="btn-prev-hotspot" class="ctrl-btn" onclick="jumpPrevHotspot()" title="Previous hotspot ([)">← hotspot</button>
          <button id="btn-next-hotspot" class="ctrl-btn" onclick="jumpNextHotspot()" title="Next hotspot (])">hotspot →</button>
        </div>
        <div class="file-nav-row">
          <div class="player-title-wrap">
            <div id="player-title" class="player-title"></div>
            <div id="player-subtitle" class="player-subtitle"></div>
          </div>
          <button id="btn-prev-file" class="ctrl-btn" onclick="jumpPrevFile()" title="Previous file">← file</button>
          <button id="btn-next-file" class="ctrl-btn" onclick="playNextInSequence()" title="Next file">file →</button>
        </div>
        <div id="hotspot-settings" class="hotspot-settings">
          <div class="hotspot-setting">
            <label for="hotspot-merge-gap">Merge gap (s)</label>
            <input id="hotspot-merge-gap" type="number" min="0" max="120" step="0.5">
          </div>
          <div class="hotspot-setting">
            <label for="hotspot-padding">Padding (s)</label>
            <input id="hotspot-padding" type="number" min="0" max="120" step="0.5">
          </div>
          <div class="hotspot-setting">
            <label for="hotspot-min-duration">Min duration (s)</label>
            <input id="hotspot-min-duration" type="number" min="1" max="180" step="0.5">
          </div>
          <div class="hotspot-setting">
            <label for="hotspot-seek-preroll">Seek preroll (s)</label>
            <input id="hotspot-seek-preroll" type="number" min="0" max="30" step="0.5">
          </div>
        </div>
        <div class="hotspot-settings-note">Live changes only affect this report view and use the detected segments already embedded in it.</div>
        <div id="hotspot-list" class="hotspot-list"></div>
      </div>
      <div class="ctrl-row">
        <button id="btn-play" class="ctrl-btn" onclick="togglePlay()" title="Play/Pause (Space)">▶</button>
        <button class="ctrl-btn" onclick="seek(-30)" title="Back 30s (Shift+←)">⏮ 30s</button>
        <button class="ctrl-btn" onclick="seek(-5)"  title="Back 5s (←)">⏪ 5s</button>
        <button class="ctrl-btn" onclick="seek(5)"   title="Fwd 5s (→)">5s ⏩</button>
        <button class="ctrl-btn" onclick="seek(30)"  title="Fwd 30s (Shift+→)">30s ⏭</button>
          <button id="btn-auto-next" class="ctrl-btn" onclick="toggleAutoNext()" title="Auto-advance through hotspots and into the next file: off" aria-pressed="false">Auto next hotspot: Off</button>
        <span class="spacer"></span>
        <button id="btn-airplay" class="ctrl-btn hidden" onclick="openAirPlayPicker()" title="AirPlay is only available in Safari/WebKit browsers that support external playback" aria-pressed="false">AirPlay</button>
        <button id="btn-mute" class="ctrl-btn" onclick="toggleMute()" title="Mute (M)">🔊</button>
        <input type="range" id="vol-track" class="vol-slider" min="0" max="1" step="0.02" value="1">
        <button class="ctrl-btn" onclick="reqFullscreen()" title="Fullscreen (F)">⛶</button>
      </div>
      <div style="margin-top:6px">
        <span class="kbd-hint">Keys: Space play/pause &bull; ←→ ±5s &bull; Shift+←→ ±30s &bull; [] hotspots &bull; ↑↓ volume &bull; M mute &bull; F fullscreen &bull; 0–9 jump &bull; Esc close</span>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = <<<DATA_JSON>>>;
<<<JS>>>
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def find_media_files(paths: List[str]) -> List[Path]:
    """Recursively find media (video or audio) files in given paths."""
    found: List[Path] = []
    for p_str in paths:
        p = Path(p_str).expanduser().resolve()
        if p.is_file():
            if p.suffix.lower() in MEDIA_EXTENSIONS:
                found.append(p)
        elif p.is_dir():
            for child in p.rglob("*"):
                if child.is_file() and child.suffix.lower() in MEDIA_EXTENSIONS:
                    found.append(child)
        else:
            print(f"WARNING: path not found: {p}", file=sys.stderr)
    return sorted(set(found))


def get_media_duration(media_path: Path) -> Optional[float]:
    """Return media (video or audio) duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(media_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return None


def extract_audio(
    video_path: Path,
    sample_rate: int,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Extract mono audio as float32 numpy array via ffmpeg.

    start_sec: seek to this position before decoding (fast keyframe seek).
    duration_sec: if given, stop after this many seconds of output audio.
    """
    cmd = ["ffmpeg"]
    if start_sec > 0:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", str(video_path)]
    if duration_sec is not None and duration_sec > 0:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd += [
        "-vn",
        "-acodec", "pcm_f32le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "f32le",
        "-loglevel", "error",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0 or not result.stdout:
            return None
        audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
        return audio
    except Exception:
        return None


def iter_audio_chunks(
    video_path: Path,
    sample_rate: int,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    chunk_seconds: float = STREAM_CHUNK_SECONDS,
) -> Iterator[np.ndarray]:
    """Yield mono float32 audio chunks from ffmpeg without loading the whole file."""
    chunk_samples = max(1, int(round(chunk_seconds * sample_rate)))
    chunk_bytes = chunk_samples * np.dtype(np.float32).itemsize

    cmd = ["ffmpeg"]
    if start_sec > 0:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", str(video_path)]
    if duration_sec is not None and duration_sec > 0:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd += [
        "-vn",
        "-acodec", "pcm_f32le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "f32le",
        "-loglevel", "error",
        "pipe:1",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    leftover = b""
    try:
        assert proc.stdout is not None
        while True:
            block = proc.stdout.read(chunk_bytes)
            if not block:
                break
            data = leftover + block
            rem = len(data) % np.dtype(np.float32).itemsize
            if rem:
                leftover = data[-rem:]
                data = data[:-rem]
            else:
                leftover = b""
            if data:
                yield np.frombuffer(data, dtype=np.float32).copy()

        stderr = b""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
        ret = proc.wait(timeout=30)
        if ret != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or f"ffmpeg exited with status {ret}")
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _rms_to_dbfs(rms: float) -> float:
    return 20.0 * math.log10(max(rms, 1.0e-9))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_segment_energy_profile(
    video_path: Path,
    sample_rate: int,
    segments: List[dict],
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> Dict[str, float]:
    """Measure positive-vs-background energy for ranked segments using a second streamed pass."""
    intervals = sorted(
        (
            (float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
            for seg in segments
            if float(seg.get("end", 0.0)) > float(seg.get("start", 0.0))
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
    if not intervals:
        return {
            "positive_rms": 0.0,
            "background_rms": 0.0,
            "positive_dbfs": -180.0,
            "background_dbfs": -180.0,
            "contrast_db": 0.0,
            "energy_confidence": 0.0,
        }

    pos_sq = 0.0
    pos_n = 0
    bg_sq = 0.0
    bg_n = 0
    consumed_samples = 0
    interval_idx = 0

    for chunk in iter_audio_chunks(
        video_path,
        sample_rate,
        start_sec=start_sec,
        duration_sec=duration_sec,
    ):
        if chunk.size == 0:
            continue

        chunk_start = start_sec + (consumed_samples / sample_rate)
        chunk_end = chunk_start + (chunk.size / sample_rate)
        consumed_samples += int(chunk.size)

        while interval_idx < len(intervals) and intervals[interval_idx][1] <= chunk_start:
            interval_idx += 1

        mask = np.zeros(chunk.size, dtype=bool)
        probe_idx = interval_idx
        while probe_idx < len(intervals):
            seg_start, seg_end = intervals[probe_idx]
            if seg_start >= chunk_end:
                break
            local_start = max(0, int(math.floor((seg_start - chunk_start) * sample_rate)))
            local_end = min(chunk.size, int(math.ceil((seg_end - chunk_start) * sample_rate)))
            if local_end > local_start:
                mask[local_start:local_end] = True
            probe_idx += 1

        if np.any(mask):
            pos_chunk = chunk[mask]
            pos_sq += float(np.dot(pos_chunk, pos_chunk))
            pos_n += int(pos_chunk.size)
        if np.any(~mask):
            bg_chunk = chunk[~mask]
            bg_sq += float(np.dot(bg_chunk, bg_chunk))
            bg_n += int(bg_chunk.size)

    positive_rms = math.sqrt(pos_sq / pos_n) if pos_n > 0 else 0.0
    background_rms = math.sqrt(bg_sq / bg_n) if bg_n > 0 else 0.0
    positive_dbfs = _rms_to_dbfs(positive_rms) if positive_rms > 0 else -180.0
    background_dbfs = _rms_to_dbfs(background_rms) if background_rms > 0 else -180.0
    contrast_db = positive_dbfs - background_dbfs if background_rms > 0 else (18.0 if positive_rms > 0 else 0.0)

    loudness_factor = _clamp01((positive_dbfs + 38.0) / 22.0)
    contrast_factor = 1.0 if background_rms <= 0 and positive_rms > 0 else _clamp01((contrast_db - 2.0) / 12.0)
    energy_confidence = _clamp01(0.65 * loudness_factor + 0.35 * contrast_factor)

    return {
        "positive_rms": round(positive_rms, 6),
        "background_rms": round(background_rms, 6),
        "positive_dbfs": round(positive_dbfs, 2),
        "background_dbfs": round(background_dbfs, 2),
        "contrast_db": round(contrast_db, 2),
        "energy_confidence": round(energy_confidence, 4),
    }


def _filter_short_runs(vocal: np.ndarray, min_frames: int) -> np.ndarray:
    """Zero out runs of True that are shorter than min_frames."""
    if min_frames <= 1 or not np.any(vocal):
        return vocal
    labeled, n = nd_label(vocal)
    if n == 0:
        return vocal
    sizes = np.bincount(labeled.ravel())   # index 0 = background
    keep  = sizes >= min_frames
    keep[0] = False
    return keep[labeled]


def _segments_from_mask(
    vocal: np.ndarray,
    frame_hop_sec: float,
    frame_span_sec: float,
    start_offset_sec: float,
    band_energy: np.ndarray,
    noise_floor: float,
    label: str,
    freq_range: Tuple[float, float],
) -> List[dict]:
    """Convert a boolean frame mask into timestamped segment metadata."""
    if vocal.size == 0 or not np.any(vocal):
        return []

    labeled, n = nd_label(vocal)
    if n == 0:
        return []

    segments: List[dict] = []
    analyzed_end = start_offset_sec + ((len(vocal) - 1) * frame_hop_sec + frame_span_sec)
    for idx in range(1, n + 1):
        pos = np.flatnonzero(labeled == idx)
        if pos.size == 0:
            continue
        start_frame = int(pos[0])
        end_frame = int(pos[-1])
        start = start_offset_sec + start_frame * frame_hop_sec
        end = min(analyzed_end, start_offset_sec + end_frame * frame_hop_sec + frame_span_sec)
        seg_energy = band_energy[pos]
        ratios = seg_energy / noise_floor if noise_floor > 0 else np.zeros_like(seg_energy)
        segments.append({
            "label": label,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
            "frame_count": int(pos.size),
            "peak_ratio": round(float(np.max(ratios)), 4) if ratios.size else 0.0,
            "avg_ratio": round(float(np.mean(ratios)), 4) if ratios.size else 0.0,
            "freq_low": float(freq_range[0]),
            "freq_high": float(freq_range[1]),
        })
    return segments


def _detect_band_activity(
    band_energy: np.ndarray,
    threshold: float,
    min_frames: int,
    noise_floor_pct: float,
    frame_hop_sec: float,
    frame_span_sec: float,
    start_offset_sec: float,
    label: str,
    freq_range: Tuple[float, float],
) -> Dict[str, object]:
    if band_energy.size == 0:
        return {"pct": 0.0, "segments": []}
    noise_floor = float(np.percentile(band_energy, noise_floor_pct))
    if noise_floor <= 0:
        return {"pct": 0.0, "segments": []}
    vocal = band_energy > threshold * noise_floor
    vocal = _filter_short_runs(vocal, min_frames)
    return {
        "pct": 100.0 * float(np.mean(vocal)),
        "segments": _segments_from_mask(
            vocal,
            frame_hop_sec,
            frame_span_sec,
            start_offset_sec,
            band_energy,
            noise_floor,
            label,
            freq_range,
        ),
    }


def analyze_vocalizations(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    female_freq: Tuple[float, float] = (350.0, 2400.0),
    male_freq: Tuple[float, float] = (100.0, 500.0),
    threshold: float = 3.0,
    min_vocal_duration: float = 0.5,
    min_audio_rms: float = 0.005,
    noise_floor_pct: float = 10.0,
) -> Tuple[float, float]:
    """
    Analyze audio for male and female vocalizations.

    Detection method:
      - Gate: if the audio RMS is below min_audio_rms the file is
        essentially silent and (0.0, 0.0) is returned immediately.
      - Compute STFT.
      - Per-band noise floor = noise_floor_pct-th percentile of that
        band's per-frame energy across all frames.  Using a low percentile
        of the *same band* means the reference adapts to whatever constant
        background (hiss, music bed, etc.) exists in that band, rather
        than being influenced by other bands.
      - A frame is 'vocal' if band_energy > threshold × band_noise_floor.
      - Short isolated runs (< min_vocal_duration seconds) are discarded.

    Returns (female_pct, male_pct) as percentages 0–100.
    """
    if audio is None or len(audio) < n_fft:
        return 0.0, 0.0

    # Silence gate: avoid false positives on near-silent files.
    # Audio from ffmpeg is normalised to [-1, 1]; 0.005 ≈ -46 dBFS.
    audio_rms = float(np.sqrt(np.mean(audio ** 2)))
    if audio_rms < min_audio_rms:
        return 0.0, 0.0

    # STFT frame rate: how many frames per second
    frames_per_sec = sample_rate / hop_length
    min_frames = max(1, int(round(min_vocal_duration * frames_per_sec)))

    # Compute STFT
    freqs, _times, Zxx = scipy_signal.stft(
        audio, fs=sample_rate,
        nperseg=n_fft, noverlap=n_fft - hop_length,
        window="hann",
    )

    mag_sq = np.abs(Zxx) ** 2  # shape: (n_freqs, n_frames)

    nyquist = sample_rate / 2.0
    # Clamp frequency bounds to valid range
    flo_f = max(0.0, min(female_freq[0], nyquist))
    fhi_f = max(0.0, min(female_freq[1], nyquist))
    flo_m = max(0.0, min(male_freq[0],   nyquist))
    fhi_m = max(0.0, min(male_freq[1],   nyquist))

    female_mask = (freqs >= flo_f) & (freqs <= fhi_f)
    male_mask   = (freqs >= flo_m) & (freqs <= fhi_m)

    def detect(freq_mask: np.ndarray) -> float:
        if not np.any(freq_mask):
            return 0.0
        band_e = np.mean(mag_sq[freq_mask], axis=0)   # per-frame energy in band
        # Low-percentile of *this band* = its noise floor.
        # Constant hiss or music bed sets this floor; vocalizations must
        # be 'threshold' times louder than that floor to be counted.
        noise_floor = float(np.percentile(band_e, noise_floor_pct))
        if noise_floor <= 0:
            return 0.0
        vocal = band_e > threshold * noise_floor
        vocal = _filter_short_runs(vocal, min_frames)
        return 100.0 * float(np.mean(vocal))

    return detect(female_mask), detect(male_mask)


def analyze_vocalizations_streaming(
    video_path: Path,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    female_freq: Tuple[float, float] = (350.0, 2400.0),
    male_freq: Tuple[float, float] = (100.0, 500.0),
    threshold: float = 3.0,
    min_vocal_duration: float = 0.5,
    min_audio_rms: float = 0.005,
    noise_floor_pct: float = 10.0,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> Dict[str, object]:
    """Stream audio from ffmpeg and compute heuristic metrics without full-file buffers."""
    if n_fft <= 0 or hop_length <= 0 or hop_length > n_fft:
        raise ValueError("n_fft and hop_length must be positive, and hop_length <= n_fft")

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    nyquist = sample_rate / 2.0
    flo_f = max(0.0, min(female_freq[0], nyquist))
    fhi_f = max(0.0, min(female_freq[1], nyquist))
    flo_m = max(0.0, min(male_freq[0], nyquist))
    fhi_m = max(0.0, min(male_freq[1], nyquist))

    female_mask = (freqs >= flo_f) & (freqs <= fhi_f)
    male_mask = (freqs >= flo_m) & (freqs <= fhi_m)
    overlap_samples = max(0, n_fft - hop_length)
    duplicate_frames = (
        (overlap_samples + hop_length - 1) // hop_length if overlap_samples > 0 else 0
    )

    total_samples = 0
    sum_squares = 0.0
    carry = np.empty(0, dtype=np.float32)
    first_chunk = True
    female_energy_chunks: List[np.ndarray] = []
    male_energy_chunks: List[np.ndarray] = []

    for chunk in iter_audio_chunks(
        video_path,
        sample_rate,
        start_sec=start_sec,
        duration_sec=duration_sec,
    ):
        if chunk.size == 0:
            continue
        total_samples += int(chunk.size)
        sum_squares += float(np.dot(chunk, chunk))

        work = np.concatenate((carry, chunk)) if carry.size else chunk
        if work.size < n_fft:
            carry = work
            first_chunk = False
            continue

        _freqs, _times, Zxx = scipy_signal.stft(
            work,
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            window="hann",
            boundary=None,
            padded=False,
        )
        mag_sq = np.abs(Zxx) ** 2
        start_col = 0 if first_chunk else duplicate_frames

        if np.any(female_mask):
            band = np.mean(mag_sq[female_mask], axis=0)
            if start_col < band.size:
                female_energy_chunks.append(band[start_col:].copy())
        if np.any(male_mask):
            band = np.mean(mag_sq[male_mask], axis=0)
            if start_col < band.size:
                male_energy_chunks.append(band[start_col:].copy())

        if overlap_samples > 0:
            carry = work[-overlap_samples:].copy()
        else:
            carry = np.empty(0, dtype=np.float32)
        first_chunk = False

    analyzed_duration = total_samples / sample_rate if total_samples > 0 else 0.0
    if total_samples < n_fft:
        return {
            "duration": analyzed_duration,
            "female_pct": 0.0,
            "male_pct": 0.0,
            "segments": [],
            "detector": "heuristic",
        }

    audio_rms = float(np.sqrt(sum_squares / total_samples)) if total_samples > 0 else 0.0
    if audio_rms < min_audio_rms:
        return {
            "duration": analyzed_duration,
            "female_pct": 0.0,
            "male_pct": 0.0,
            "segments": [],
            "detector": "heuristic",
        }

    frames_per_sec = sample_rate / hop_length
    frame_hop_sec = hop_length / sample_rate
    frame_span_sec = n_fft / sample_rate
    min_frames = max(1, int(round(min_vocal_duration * frames_per_sec)))

    female_energy = (
        np.concatenate(female_energy_chunks) if female_energy_chunks else np.empty(0, dtype=np.float32)
    )
    male_energy = (
        np.concatenate(male_energy_chunks) if male_energy_chunks else np.empty(0, dtype=np.float32)
    )

    female_result = _detect_band_activity(
        female_energy,
        threshold,
        min_frames,
        noise_floor_pct,
        frame_hop_sec,
        frame_span_sec,
        start_sec,
        "female",
        female_freq,
    )
    male_result = _detect_band_activity(
        male_energy,
        threshold,
        min_frames,
        noise_floor_pct,
        frame_hop_sec,
        frame_span_sec,
        start_sec,
        "male",
        male_freq,
    )

    return {
        "duration": analyzed_duration,
        "female_pct": float(female_result["pct"]),
        "male_pct": float(male_result["pct"]),
        "segments": list(female_result["segments"]) + list(male_result["segments"]),
        "detector": "heuristic",
    }


_YAMNET_RESOURCES: Dict[str, object] = {}
DEFAULT_TFHUB_CACHE_DIR = Path.home() / ".cache" / "tfhub_modules"


def _yamnet_saved_model_marker(path: Path) -> Optional[Path]:
    for name in ("saved_model.pb", "saved_model.pbtxt"):
        marker = path / name
        if marker.exists():
            return marker
    return None


def _yamnet_cache_candidates(model_handle: str) -> List[Path]:
    handles = [model_handle]
    if "tf-hub-format=" not in model_handle:
        sep = "&" if "?" in model_handle else "?"
        handles.append(f"{model_handle}{sep}tf-hub-format=compressed")

    cache_roots: List[Path] = []
    env_cache = os.environ.get("TFHUB_CACHE_DIR", "").strip()
    if env_cache:
        cache_roots.append(Path(env_cache))
    cache_roots.append(DEFAULT_TFHUB_CACHE_DIR)
    cache_roots.append(Path(tempfile.gettempdir()) / "tfhub_modules")

    seen: set[str] = set()
    candidates: List[Path] = []
    for root in cache_roots:
        for handle in handles:
            candidate = root / hashlib.sha1(handle.encode("utf-8")).hexdigest()
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _load_yamnet_resources(model_handle: str) -> Dict[str, object]:
    cached = _YAMNET_RESOURCES.get(model_handle)
    if isinstance(cached, dict):
        return cached

    try:
        import tensorflow as tf  # type: ignore[import-not-found]
        import tensorflow_hub as hub  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "YAMNet detector requires 'tensorflow' and 'tensorflow-hub'. "
            "Install them before using --detector yamnet."
        ) from exc

    if not os.environ.get("TFHUB_CACHE_DIR"):
        os.environ["TFHUB_CACHE_DIR"] = str(DEFAULT_TFHUB_CACHE_DIR)

    try:
        model = hub.load(model_handle)
    except Exception as exc:
        valid_cache_hits: List[Path] = []
        incomplete_cache_hits: List[Path] = []
        for candidate in _yamnet_cache_candidates(model_handle):
            if not candidate.exists():
                continue
            if _yamnet_saved_model_marker(candidate):
                valid_cache_hits.append(candidate)
                continue
            incomplete_cache_hits.append(candidate)

        for candidate in valid_cache_hits:
            try:
                model = hub.load(str(candidate))
                break
            except Exception:
                continue
        else:
            checked = ", ".join(str(path) for path in _yamnet_cache_candidates(model_handle))
            extra = ""
            if incomplete_cache_hits:
                extra = (
                    " Found incomplete cached download(s) at: "
                    + ", ".join(str(path) for path in incomplete_cache_hits)
                    + "."
                )
            raise RuntimeError(
                f"Could not load YAMNet model from {model_handle!r}. "
                f"Checked TF-Hub cache paths: {checked}.{extra} "
                f"Pass a local SavedModel path with --yamnet-model, or download the model once "
                f"with network access so it is cached under {os.environ['TFHUB_CACHE_DIR']}."
            ) from exc

    class_map_value = model.class_map_path().numpy()
    class_map_path = class_map_value.decode("utf-8") if isinstance(class_map_value, bytes) else str(class_map_value)
    class_names: List[str] = []
    with open(class_map_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            class_names.append(row["display_name"])

    resources = {
        "tf": tf,
        "model": model,
        "class_names": class_names,
        "class_index": {name: idx for idx, name in enumerate(class_names)},
    }
    _YAMNET_RESOURCES[model_handle] = resources
    return resources


def _yamnet_weighted_score(
    frame_scores: np.ndarray,
    class_index: Dict[str, int],
    positive_weights: Dict[str, float],
    negative_weights: Dict[str, float],
) -> np.ndarray:
    if frame_scores.size == 0:
        return np.empty(0, dtype=np.float32)

    pos = np.zeros(frame_scores.shape[0], dtype=np.float32)
    neg = np.zeros(frame_scores.shape[0], dtype=np.float32)
    for label, weight in positive_weights.items():
        idx = class_index.get(label)
        if idx is not None:
            pos += frame_scores[:, idx].astype(np.float32) * weight
    for label, weight in negative_weights.items():
        idx = class_index.get(label)
        if idx is not None:
            neg += frame_scores[:, idx].astype(np.float32) * weight
    return np.clip(pos - neg, 0.0, 1.0)


def _segments_from_scores(
    scores: np.ndarray,
    label: str,
    threshold: float,
    frame_hop_sec: float,
    frame_span_sec: float,
    start_offset_sec: float,
) -> Tuple[float, List[dict]]:
    if scores.size == 0:
        return 0.0, []
    active = scores >= threshold
    if not np.any(active):
        return 0.0, []

    labeled, n = nd_label(active)
    segments: List[dict] = []
    analyzed_end = start_offset_sec + ((len(scores) - 1) * frame_hop_sec + frame_span_sec)
    for idx in range(1, n + 1):
        pos = np.flatnonzero(labeled == idx)
        if pos.size == 0:
            continue
        start_frame = int(pos[0])
        end_frame = int(pos[-1])
        start = start_offset_sec + start_frame * frame_hop_sec
        end = min(analyzed_end, start_offset_sec + end_frame * frame_hop_sec + frame_span_sec)
        seg_scores = scores[pos]
        segments.append({
            "label": label,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
            "frame_count": int(pos.size),
            "peak_score": round(float(np.max(seg_scores)), 4),
            "avg_score": round(float(np.mean(seg_scores)), 4),
        })
    return 100.0 * float(np.mean(active)), segments


def _yamnet_top_labels(
    row_scores: np.ndarray,
    class_names: List[str],
    top_k: int,
) -> List[dict]:
    if row_scores.size == 0 or top_k <= 0:
        return []
    k = min(top_k, row_scores.size)
    idxs = np.argpartition(row_scores, -k)[-k:]
    idxs = idxs[np.argsort(row_scores[idxs])[::-1]]
    return [
        {"label": class_names[int(idx)], "score": round(float(row_scores[int(idx)]), 4)}
        for idx in idxs
    ]


def _window_rms_envelope(
    audio: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> np.ndarray:
    if audio.size < window_samples or window_samples <= 0 or hop_samples <= 0:
        return np.empty(0, dtype=np.float32)
    squared = np.square(audio.astype(np.float64, copy=False))
    cumsum = np.empty(squared.size + 1, dtype=np.float64)
    cumsum[0] = 0.0
    np.cumsum(squared, out=cumsum[1:])
    starts = np.arange(0, audio.size - window_samples + 1, hop_samples, dtype=np.int64)
    ends = starts + window_samples
    power = (cumsum[ends] - cumsum[starts]) / float(window_samples)
    return np.sqrt(power).astype(np.float32)


def _adaptive_rms_thresholds(
    frame_rms: np.ndarray,
    history: List[float],
    min_window_rms: float,
    context_frames: int,
    context_ratio: float,
) -> np.ndarray:
    thresholds = np.empty(frame_rms.shape[0], dtype=np.float32)
    for idx, rms in enumerate(frame_rms):
        threshold = float(min_window_rms)
        if context_ratio > 0 and history:
            if context_frames > 0 and len(history) > context_frames:
                context = history[-context_frames:]
            else:
                context = history
            context_ref = float(np.percentile(context, YAMNET_CONTEXT_RMS_PERCENTILE))
            threshold = max(threshold, context_ref * context_ratio)
        thresholds[idx] = threshold
        history.append(float(rms))
    return thresholds


def analyze_yamnet_streaming(
    video_path: Path,
    model_handle: str,
    score_threshold: float,
    min_window_rms: float = YAMNET_MIN_WINDOW_RMS,
    context_rms_seconds: float = YAMNET_CONTEXT_RMS_SECONDS,
    context_rms_ratio: float = YAMNET_CONTEXT_RMS_RATIO,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    collect_debug: bool = False,
    top_k: int = 8,
) -> Dict[str, object]:
    """Run YAMNet on streaming audio chunks and emit scream/moan segments."""
    resources = _load_yamnet_resources(model_handle)
    tf = resources["tf"]  # type: ignore[assignment]
    model = resources["model"]
    class_index = resources["class_index"]  # type: ignore[assignment]
    class_names = resources["class_names"]  # type: ignore[assignment]

    frame_hop_sec = YAMNET_PATCH_HOP_SECONDS
    frame_span_sec = YAMNET_PATCH_WINDOW_SECONDS
    frame_span_samples = int(round(frame_span_sec * YAMNET_SAMPLE_RATE))
    frame_hop_samples = int(round(frame_hop_sec * YAMNET_SAMPLE_RATE))
    context_frames = max(0, int(round(context_rms_seconds / frame_hop_sec)))
    overlap_samples = int(round(YAMNET_CHUNK_OVERLAP_SECONDS * YAMNET_SAMPLE_RATE))
    duplicate_frames = int(np.floor(YAMNET_CHUNK_OVERLAP_SECONDS / frame_hop_sec))

    total_samples = 0
    carry = np.empty(0, dtype=np.float32)
    first_chunk = True
    emitted_frames = 0
    rms_history: List[float] = []
    vocal_scores_chunks: List[np.ndarray] = []
    debug_windows: List[dict] = []

    for chunk in iter_audio_chunks(
        video_path,
        YAMNET_SAMPLE_RATE,
        start_sec=start_sec,
        duration_sec=duration_sec,
        chunk_seconds=YAMNET_CHUNK_SECONDS,
    ):
        if chunk.size == 0:
            continue
        total_samples += int(chunk.size)
        work = np.concatenate((carry, chunk)) if carry.size else chunk
        if work.size < frame_span_samples:
            carry = work
            first_chunk = False
            continue

        frame_rms = _window_rms_envelope(work, frame_span_samples, frame_hop_samples)
        start_row = 0 if first_chunk else min(duplicate_frames, frame_rms.shape[0])
        local_frame_rms = frame_rms[start_row:]
        if local_frame_rms.size == 0:
            if overlap_samples > 0 and work.size > overlap_samples:
                carry = work[-overlap_samples:].copy()
            else:
                carry = work.copy()
            first_chunk = False
            continue

        rms_thresholds = _adaptive_rms_thresholds(
            local_frame_rms,
            rms_history,
            min_window_rms=min_window_rms,
            context_frames=context_frames,
            context_ratio=context_rms_ratio,
        )
        energy_mask = local_frame_rms >= rms_thresholds
        if not np.any(energy_mask):
            vocal_scores_chunks.append(np.zeros(local_frame_rms.shape[0], dtype=np.float32))
            if collect_debug:
                for local_idx, audio_rms in enumerate(local_frame_rms):
                    frame_idx = emitted_frames + local_idx
                    win_start = start_sec + frame_idx * frame_hop_sec
                    debug_windows.append({
                        "start": round(win_start, 3),
                        "end": round(win_start + frame_span_sec, 3),
                        "audio_rms": round(float(audio_rms), 6),
                        "audio_rms_threshold": round(float(rms_thresholds[local_idx]), 6),
                        "energy_gated": True,
                        "vocalization_score": 0.0,
                        "top_labels": [],
                    })
            emitted_frames += local_frame_rms.shape[0]
            if overlap_samples > 0 and work.size > overlap_samples:
                carry = work[-overlap_samples:].copy()
            else:
                carry = work.copy()
            first_chunk = False
            continue

        waveform = tf.convert_to_tensor(work, dtype=tf.float32)
        scores, _embeddings, _spectrogram = model(waveform)
        frame_scores = scores.numpy()
        start_row = 0 if first_chunk else min(duplicate_frames, frame_scores.shape[0])
        if start_row < frame_scores.shape[0]:
            chunk_frame_scores = frame_scores[start_row:]
            row_count = min(chunk_frame_scores.shape[0], local_frame_rms.shape[0])
            chunk_frame_scores = chunk_frame_scores[:row_count]
            local_frame_rms = local_frame_rms[:row_count]
            rms_thresholds = rms_thresholds[:row_count]
            energy_mask = energy_mask[:row_count]
            vocal_chunk_scores = _yamnet_weighted_score(
                chunk_frame_scores,
                class_index,
                YAMNET_VOCALIZATION_WEIGHTS,
                YAMNET_NEGATIVE_WEIGHTS,
            )
            if min_window_rms > 0:
                vocal_chunk_scores = np.where(
                    energy_mask,
                    vocal_chunk_scores,
                    np.zeros_like(vocal_chunk_scores),
                )
            vocal_scores_chunks.append(vocal_chunk_scores)
            if collect_debug:
                for local_idx, row_scores in enumerate(chunk_frame_scores):
                    frame_idx = emitted_frames + local_idx
                    win_start = start_sec + frame_idx * frame_hop_sec
                    energy_gated = bool(not energy_mask[local_idx])
                    debug_windows.append({
                        "start": round(win_start, 3),
                        "end": round(win_start + frame_span_sec, 3),
                        "audio_rms": round(float(local_frame_rms[local_idx]), 6),
                        "audio_rms_threshold": round(float(rms_thresholds[local_idx]), 6),
                        "energy_gated": energy_gated,
                        "vocalization_score": round(float(vocal_chunk_scores[local_idx]), 4),
                        "top_labels": [] if energy_gated else _yamnet_top_labels(row_scores, class_names, top_k),
                    })
            emitted_frames += chunk_frame_scores.shape[0]

        if overlap_samples > 0 and work.size > overlap_samples:
            carry = work[-overlap_samples:].copy()
        else:
            carry = work.copy()
        first_chunk = False

    analyzed_duration = total_samples / YAMNET_SAMPLE_RATE if total_samples > 0 else 0.0
    vocal_scores = (
        np.concatenate(vocal_scores_chunks) if vocal_scores_chunks else np.empty(0, dtype=np.float32)
    )

    vocal_pct, vocal_segments = _segments_from_scores(
        vocal_scores,
        "vocalization",
        score_threshold,
        frame_hop_sec,
        frame_span_sec,
        start_sec,
    )

    return {
        "duration": analyzed_duration,
        "female_pct": vocal_pct,
        "male_pct": 0.0,
        "segments": vocal_segments,
        "detector": "yamnet",
        "yamnet_debug": {
            "frame_hop_sec": frame_hop_sec,
            "frame_span_sec": frame_span_sec,
            "score_threshold": score_threshold,
            "min_window_rms": min_window_rms,
            "context_rms_seconds": context_rms_seconds,
            "context_rms_ratio": context_rms_ratio,
            "windows": debug_windows,
        } if collect_debug else None,
    }


def analyze_media_file(
    video_path: Path,
    detector: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    female_freq: Tuple[float, float],
    male_freq: Tuple[float, float],
    threshold: float,
    min_vocal_duration: float,
    min_audio_rms: float,
    noise_floor_pct: float,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    yamnet_model: str = "https://tfhub.dev/google/yamnet/1",
    yamnet_score_threshold: float = 0.05,
    yamnet_min_window_rms: float = YAMNET_MIN_WINDOW_RMS,
    yamnet_context_rms_seconds: float = YAMNET_CONTEXT_RMS_SECONDS,
    yamnet_context_rms_ratio: float = YAMNET_CONTEXT_RMS_RATIO,
    yamnet_collect_debug: bool = False,
    yamnet_top_k: int = 8,
) -> Dict[str, object]:
    """Detector abstraction point for current heuristic and future model-based detectors."""
    if detector == "heuristic":
        return analyze_vocalizations_streaming(
            video_path,
            sample_rate,
            n_fft,
            hop_length,
            female_freq,
            male_freq,
            threshold,
            min_vocal_duration,
            min_audio_rms,
            noise_floor_pct,
            start_sec=start_sec,
            duration_sec=duration_sec,
        )
    if detector == "yamnet":
        return analyze_yamnet_streaming(
            video_path,
            model_handle=yamnet_model,
            score_threshold=yamnet_score_threshold,
            min_window_rms=yamnet_min_window_rms,
            context_rms_seconds=yamnet_context_rms_seconds,
            context_rms_ratio=yamnet_context_rms_ratio,
            start_sec=start_sec,
            duration_sec=duration_sec,
            collect_debug=yamnet_collect_debug,
            top_k=yamnet_top_k,
        )
    raise ValueError(f"Unsupported detector: {detector}")


def export_segments_json(results: List[dict], output_path: Path) -> None:
    payload = {
        "version": 1,
        "files": [
            {
                "path": r["path"],
                "name": r["name"],
                "kind": r["kind"],
                "duration": round(float(r["duration"] or 0.0), 3),
                "female_pct": round(float(max(r.get("female_pct", -1.0), 0.0)), 3),
                "male_pct": round(float(max(r.get("male_pct", -1.0), 0.0)), 3),
                "detector": r.get("detector", "heuristic"),
                "metric_labels": [label for label in detector_metric_labels(str(r.get("detector", "heuristic"))) if label],
                "segments": r.get("segments", []),
                "hotspots": r.get("hotspots", []),
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_yamnet_debug_json(results: List[dict], output_path: Path) -> None:
    payload = {
        "version": 1,
        "files": [
            {
                "path": r["path"],
                "name": r["name"],
                "kind": r["kind"],
                "duration": round(float(r["duration"] or 0.0), 3),
                "detector": r.get("detector", "heuristic"),
                "metric_labels": [label for label in detector_metric_labels(str(r.get("detector", "heuristic"))) if label],
                "yamnet_debug": r.get("yamnet_debug"),
            }
            for r in results
            if r.get("detector") == "yamnet" and r.get("yamnet_debug")
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "?"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(path: Path, params: Dict[str, object]) -> str:
    """Unique cache key: file identity + all analysis parameters.

    Including params means any change to threshold, frequency ranges, etc.
    automatically invalidates cached results for that file.
    """
    st = path.stat()
    param_str = json.dumps(params, sort_keys=True)
    return f"{path}|{st.st_mtime:.3f}|{st.st_size}|{param_str}"


def load_cache(cache_path: Path) -> Dict[str, dict]:
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(cache_path: Path, cache: Dict[str, dict]) -> None:
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"WARNING: could not save cache: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def _analyze_worker(video_path_str: str, params: Dict[str, object]) -> Dict[str, object]:
    """Pure-computation worker: extract audio and analyze; no cache I/O.

    Safe to run inside a ProcessPoolExecutor worker process.
    Returns a dict with keys: duration, female_pct, male_pct.
    """
    video_path      = Path(video_path_str)
    sample_rate     = int(params["analysis_sample_rate"])  # type: ignore[arg-type]
    n_fft           = int(params["n_fft"])            # type: ignore[arg-type]
    hop_length      = int(params["hop_length"])       # type: ignore[arg-type]
    female_freq     = tuple(params["female_freq"])    # type: ignore[arg-type]
    male_freq       = tuple(params["male_freq"])      # type: ignore[arg-type]
    threshold       = float(params["threshold"])      # type: ignore[arg-type]
    min_voc_dur     = float(params["min_vocal_duration"])  # type: ignore[arg-type]
    min_audio_rms   = float(params["min_audio_rms"]) # type: ignore[arg-type]
    noise_floor_pct = float(params["noise_floor_pct"])  # type: ignore[arg-type]
    clip_duration   = float(params["clip_duration"]) # type: ignore[arg-type]
    detector        = str(params["detector"])        # type: ignore[arg-type]
    yamnet_model    = str(params["yamnet_model"])    # type: ignore[arg-type]
    yamnet_score_threshold = float(params["yamnet_score_threshold"])  # type: ignore[arg-type]
    yamnet_min_window_rms = float(params["yamnet_min_window_rms"])  # type: ignore[arg-type]
    yamnet_context_rms_seconds = float(params["yamnet_context_rms_seconds"])  # type: ignore[arg-type]
    yamnet_context_rms_ratio = float(params["yamnet_context_rms_ratio"])  # type: ignore[arg-type]
    yamnet_collect_debug = bool(params["yamnet_collect_debug"])  # type: ignore[arg-type]
    yamnet_top_k = int(params["yamnet_top_k"])  # type: ignore[arg-type]

    # When a clip duration is requested, seek to the end of the file.
    # We need the full duration from ffprobe to compute the start offset.
    start_sec: float = 0.0
    extract_dur: Optional[float] = None
    reported_duration: Optional[float] = None   # full file duration when clipping
    if clip_duration > 0:
        full_dur = get_media_duration(video_path) or 0.0
        reported_duration = full_dur
        if full_dur > clip_duration:
            start_sec   = full_dur - clip_duration
            extract_dur = clip_duration
        # else: file is shorter than clip_duration — analyze it entirely

    try:
        analysis = analyze_media_file(
            video_path,
            detector,
            sample_rate,
            n_fft,
            hop_length,
            female_freq,
            male_freq,
            threshold,
            min_voc_dur,
            min_audio_rms,
            noise_floor_pct,
            start_sec=start_sec,
            duration_sec=extract_dur,
            yamnet_model=yamnet_model,
            yamnet_score_threshold=yamnet_score_threshold,
            yamnet_min_window_rms=yamnet_min_window_rms,
            yamnet_context_rms_seconds=yamnet_context_rms_seconds,
            yamnet_context_rms_ratio=yamnet_context_rms_ratio,
            yamnet_collect_debug=yamnet_collect_debug,
            yamnet_top_k=yamnet_top_k,
        )
    except Exception:
        if detector != "heuristic":
            raise
        # Audio extraction failed; fall back to ffprobe for duration.
        duration = reported_duration or get_media_duration(video_path) or 0.0
        return {
            "duration": duration,
            "female_pct": -1.0,
            "male_pct": -1.0,
            "segments": [],
            "detector": detector,
            "yamnet_debug": None,
            "positive_rms": 0.0,
            "background_rms": 0.0,
            "positive_dbfs": -180.0,
            "background_dbfs": -180.0,
            "contrast_db": 0.0,
            "energy_confidence": 0.0,
        }

    # Use ffprobe-reported full duration when clipping; otherwise derive from analyzed audio.
    duration = reported_duration if reported_duration is not None else float(analysis["duration"])
    analysis["duration"] = duration
    try:
        analysis.update(
            compute_segment_energy_profile(
                video_path,
                sample_rate,
                list(analysis.get("segments", [])),  # type: ignore[arg-type]
                start_sec=start_sec,
                duration_sec=extract_dur,
            )
        )
    except Exception:
        analysis.update({
            "positive_rms": 0.0,
            "background_rms": 0.0,
            "positive_dbfs": -180.0,
            "background_dbfs": -180.0,
            "contrast_db": 0.0,
            "energy_confidence": 0.0,
        })
    return analysis


def _make_result(video_path: Path, data: Dict[str, object], cached: bool) -> dict:
    """Build the per-file result dict from worker output."""
    duration = data.get("duration")
    segments = list(data.get("segments", []))  # type: ignore[arg-type]
    return {
        "path":       str(video_path),
        "url":        video_path.as_uri(),
        "name":       video_path.name,
        "kind":       media_kind(video_path),
        "duration":   duration,
        "female_pct": data.get("female_pct", -1.0),
        "male_pct":   data.get("male_pct",   -1.0),
        "energy_confidence": data.get("energy_confidence", 0.0),
        "positive_dbfs": data.get("positive_dbfs", -180.0),
        "background_dbfs": data.get("background_dbfs", -180.0),
        "contrast_db": data.get("contrast_db", 0.0),
        "segments":   segments,
        "hotspots":   build_hotspots(segments, duration if isinstance(duration, (int, float)) else None),
        "detector":   data.get("detector", "heuristic"),
        "yamnet_debug": data.get("yamnet_debug"),
        "cached":     cached,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(results: List[dict], args: argparse.Namespace) -> str:
    max_duration = max((r["duration"] or 0 for r in results), default=1.0) or 1.0
    metric1_label, metric2_label = detector_metric_labels(args.detector)
    has_second_metric = metric2_label is not None

    js_data = []
    for r in results:
        dur = r["duration"] or 0
        segments = list(r.get("segments", []))
        parent_dir = Path(r["path"]).parent.name or str(Path(r["path"]).parent)
        positive_duration = _union_duration(
            sorted(
                (
                    (float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
                    for seg in segments
                    if float(seg.get("end", 0.0)) > float(seg.get("start", 0.0))
                ),
                key=lambda pair: (pair[0], pair[1]),
            )
        )
        js_data.append({
            "name":         r["name"],
            "parent_dir":   parent_dir,
            "path":         r["path"],
            "url":          r["url"],
            "kind":         r.get("kind", "video"),
            "duration":     round(dur, 3),
            "duration_fmt": format_duration(r["duration"]),
            "positive_duration": round(positive_duration, 3),
            "female_pct":   round(max(r["female_pct"], 0), 2),
            "male_pct":     round(max(r["male_pct"],   0), 2),
            "energy_confidence": round(float(max(min(r.get("energy_confidence", 0.0), 1.0), 0.0)), 4),
            "positive_dbfs": round(float(r.get("positive_dbfs", -180.0)), 2),
            "background_dbfs": round(float(r.get("background_dbfs", -180.0)), 2),
            "contrast_db": round(float(r.get("contrast_db", 0.0)), 2),
            "segments":     segments,
            "hotspots":     r.get("hotspots", []),
        })

    data_json = json.dumps(js_data, ensure_ascii=True)
    js = (
        JS.replace("HAS_SECOND_METRIC", "true" if has_second_metric else "false")
        .replace("<<<HOTSPOT_MERGE_GAP>>>", json.dumps(HOTSPOT_MERGE_GAP_SECONDS))
        .replace("<<<HOTSPOT_PADDING>>>", json.dumps(HOTSPOT_PADDING_SECONDS))
        .replace("<<<HOTSPOT_MIN_DURATION>>>", json.dumps(HOTSPOT_MIN_DURATION_SECONDS))
        .replace("<<<HOTSPOT_SEEK_PREROLL>>>", json.dumps(HOTSPOT_SEEK_PREROLL_SECONDS))
    )

    html = (
        HTML_TEMPLATE
        .replace("<<<CSS>>>",       CSS)
        .replace("<<<JS>>>",        js)
        .replace("<<<DATA_JSON>>>", data_json)
        .replace("<<<FILE_COUNT>>>", str(len(results)))
        .replace("<<<SUBTITLE>>>", build_report_subtitle(len(results), args))
        .replace("<<<METRIC1_LABEL>>>", metric1_label)
        .replace("<<<METRIC2_LABEL>>>", metric2_label or "")
        .replace("<<<METRIC2_GROUP_CLASS>>>", "" if has_second_metric else "hidden-metric")
        .replace("<<<METRIC2_COL_CLASS>>>", "" if has_second_metric else "hidden-metric")
    )
    return html


# ---------------------------------------------------------------------------
# Local report server (Safari-friendly media playback)
# ---------------------------------------------------------------------------

_MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".ogv": "video/ogg",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
}


def path_from_file_url(url: str) -> Path:
    """Convert a file:// URL to a filesystem Path."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    # urlparse("file:///C:/x") on Windows yields "/C:/x"; strip the extra slash.
    if sys.platform == "win32" and re.match(r"^/[A-Za-z]:/", path):
        path = path[1:]
    return Path(path)


def guess_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _MIME_BY_EXT:
        return _MIME_BY_EXT[ext]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def extract_media_paths_from_report(html: str) -> List[Path]:
    """Pull per-row filesystem paths out of a generated report's DATA blob."""
    match = re.search(r"const DATA\s*=\s*(\[.*?\]);", html, re.S)
    if not match:
        raise ValueError("Could not find DATA array in report HTML")
    data = json.loads(match.group(1))
    paths: List[Path] = []
    for item in data:
        raw_path = item.get("path")
        if raw_path:
            paths.append(Path(str(raw_path)))
            continue
        url = str(item.get("url") or "")
        if url.startswith("file:"):
            paths.append(path_from_file_url(url))
            continue
        raise ValueError(
            f"Report entry {item.get('name')!r} has no filesystem path; "
            "re-generate the report with a current screamfinder.py"
        )
    return paths


def _parse_byte_range(range_header: str, size: int) -> Optional[Tuple[int, int]]:
    """Parse a single-range Range header into inclusive (start, end) byte indexes."""
    if not range_header or "," in range_header:
        return None
    match = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
    if not match:
        return None
    start_s, end_s = match.group(1), match.group(2)
    if start_s == "" and end_s == "":
        return None
    if start_s == "":
        # suffix range: last N bytes
        length = int(end_s)
        if length <= 0:
            return None
        start = max(size - length, 0)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
        if start >= size:
            return None
        end = min(end, size - 1)
        if end < start:
            return None
    return start, end


def rewrite_report_media_urls(html: str) -> str:
    """Point DATA[].url at /media/<idx> so file:// URLs are not used over HTTP."""
    match = re.search(r"const DATA\s*=\s*(\[.*?\]);", html, re.S)
    if not match:
        return html
    data = json.loads(match.group(1))
    for idx, item in enumerate(data):
        item["url"] = f"/media/{idx}"
    return html[: match.start(1)] + json.dumps(data, ensure_ascii=True) + html[match.end(1) :]


class _ReportHandler(BaseHTTPRequestHandler):
    """Serve the HTML report and allowlisted media with HTTP Range support."""

    html_path: Path = Path(".")
    media_paths: Sequence[Path] = ()

    def log_message(self, fmt: str, *args: object) -> None:
        # Keep the terminal quiet unless something fails.
        if args and len(args) >= 2 and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle(send_body=False)

    def do_GET(self) -> None:  # noqa: N802
        self._handle(send_body=True)

    def _handle(self, send_body: bool) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html", "/report.html"):
            html = rewrite_report_media_urls(
                self.html_path.read_text(encoding="utf-8")
            )
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if send_body:
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    return
            return
        if path.startswith("/media/"):
            token = path[len("/media/"):]
            if not token.isdigit():
                self.send_error(404, "Not Found")
                return
            idx = int(token)
            if idx < 0 or idx >= len(self.media_paths):
                self.send_error(404, "Not Found")
                return
            media = self.media_paths[idx]
            if not media.is_file():
                self.send_error(404, f"Missing media: {media.name}")
                return
            self._send_media(media, send_body)
            return
        self.send_error(404, "Not Found")

    def _send_media(self, file_path: Path, send_body: bool) -> None:
        size = file_path.stat().st_size
        content_type = guess_media_type(file_path)
        range_header = self.headers.get("Range")
        byte_range = _parse_byte_range(range_header, size) if range_header else None

        if range_header and byte_range is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        if byte_range is None:
            start, end = 0, size - 1 if size else 0
            status = 200
        else:
            start, end = byte_range
            status = 206

        length = (end - start + 1) if size else 0
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "private, max-age=0")
        self.end_headers()
        if not send_body or length <= 0:
            return
        try:
            with file_path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return


def open_report_url(url: str, browser: str = "default") -> None:
    """Open a report URL in the requested browser (macOS prefers `open -a`)."""
    browser = (browser or "default").strip().lower()
    if browser in ("", "default", "auto"):
        webbrowser.open(url)
        return
    if browser in ("none", "no", "off"):
        return

    app_names = {
        "safari": "Safari",
        "chrome": "Google Chrome",
        "chromium": "Chromium",
        "firefox": "Firefox",
        "edge": "Microsoft Edge",
        "brave": "Brave Browser",
    }
    app = app_names.get(browser, browser)

    if sys.platform == "darwin":
        try:
            subprocess.run(["open", "-a", app, url], check=True)
            return
        except (OSError, subprocess.CalledProcessError) as exc:
            print(
                f"WARNING: could not open {app!r} ({exc}); falling back to default browser.",
                file=sys.stderr,
            )
            webbrowser.open(url)
            return

    # Non-macOS: try the named controller, then PATH binary, then default.
    try:
        controller = webbrowser.get(browser)
        controller.open(url)
        return
    except webbrowser.Error:
        pass
    binary = shutil.which(browser)
    if binary:
        try:
            subprocess.Popen([binary, url])  # noqa: S603
            return
        except OSError as exc:
            print(
                f"WARNING: could not launch {binary!r} ({exc}); falling back to default browser.",
                file=sys.stderr,
            )
    webbrowser.open(url)


def serve_report(
    html_path: Path,
    media_paths: Sequence[Path],
    port: int = 8765,
    browser: str = "default",
) -> None:
    """Serve a report over localhost so Safari can play media outside the report folder."""
    html_path = html_path.resolve()
    if not html_path.is_file():
        raise FileNotFoundError(f"Report not found: {html_path}")

    handler = type(
        "BoundReportHandler",
        (_ReportHandler,),
        {
            "html_path": html_path,
            "media_paths": list(media_paths),
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Serving report at {url}", file=sys.stderr)
    print("Press Ctrl+C to stop.", file=sys.stderr)
    open_report_url(url, browser=browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, object]:
    """Load TOML config file and return a dict of argparse defaults.

    CLI arguments always take precedence over values in the config file.
    """
    if not config_path.exists():
        return {}
    if tomllib is None:
        print(
            f"WARNING: config file '{config_path}' found but tomllib is not available "
            "(Python < 3.11 and tomli not installed). Skipping.",
            file=sys.stderr,
        )
        return {}
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        print(f"WARNING: could not read config '{config_path}': {e}", file=sys.stderr)
        return {}

    defaults: Dict[str, object] = {}
    str_keys   = (
        "output",
        "cache",
        "segments_json",
        "detector",
        "yamnet_model",
        "yamnet_label_debug_json",
        "browser",
    )
    float_keys = (
        "threshold",
        "clip_duration",
        "min_vocal_duration",
        "min_audio_rms",
        "noise_floor_pct",
        "yamnet_score_threshold",
        "yamnet_min_window_rms",
        "yamnet_context_rms_seconds",
        "yamnet_context_rms_ratio",
    )
    int_keys   = ("sample_rate", "jobs", "n_fft", "hop_length", "yamnet_top_k", "port")
    bool_keys  = ("no_cache", "force", "serve")

    for k in str_keys:
        if k in data:
            defaults[k] = str(data[k])
    for k in float_keys:
        if k in data:
            defaults[k] = float(data[k])
    for k in int_keys:
        if k in data:
            defaults[k] = int(data[k])
    for k in bool_keys:
        if k in data:
            defaults[k] = bool(data[k])
    for key in ("female_freq", "male_freq"):
        if key in data:
            val = data[key]
            if isinstance(val, (list, tuple)) and len(val) == 2:
                defaults[key] = [float(val[0]), float(val[1])]

    return defaults


def _parse_duration(s: str) -> float:
    """Parse a duration string: plain seconds (float) or [h:]mm:ss."""
    s = s.strip()
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        pass
    raise argparse.ArgumentTypeError(
        f"Invalid duration {s!r} — use seconds (e.g. 300) or [h:]mm:ss (e.g. 5:00)"
    )


def check_dependency(name: str) -> None:
    if not shutil.which(name):
        print(f"ERROR: '{name}' not found in PATH. Install ffmpeg and try again.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    # ── Step 1: find --config before the full parse ───────────────────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="screamfinder.toml")
    pre_args, _ = pre.parse_known_args()

    config_defaults = load_config(Path(pre_args.config))

    # ── Step 2: main parser ───────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Analyze audio and video files for audible vocalizations and generate an HTML report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="*", metavar="PATH",
        help="Media file(s) (video or audio) and/or directory(s) to analyze",
    )
    parser.add_argument(
        "--config", default="screamfinder.toml", metavar="FILE",
        help="TOML config file (CLI args override values from this file)",
    )
    parser.add_argument(
        "-o", "--output", default="screamfinder.html", metavar="FILE",
        help="Output HTML file",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="After writing the report, serve it on localhost and open a browser "
             "(required for Safari when media lives outside the report folder)",
    )
    parser.add_argument(
        "--serve-report", metavar="FILE", default="",
        help="Serve an existing HTML report on localhost (no re-analysis) and open a browser",
    )
    parser.add_argument(
        "--port", type=int, default=8765, metavar="N",
        help="Port for --serve / --serve-report",
    )
    parser.add_argument(
        "--browser", default="default", metavar="NAME",
        help="Browser to open with --serve / --serve-report: default, safari, chrome, "
             "firefox, edge, brave, or none",
    )
    parser.add_argument(
        "--segments-json", default="", metavar="FILE",
        help="Optional JSON export file for per-file segment timestamps and metadata",
    )
    parser.add_argument(
        "--detector", choices=DETECTOR_CHOICES, default="heuristic",
        help="Detection backend to use. 'heuristic' is the current streaming STFT detector.",
    )
    parser.add_argument(
        "--yamnet-model", default="https://tfhub.dev/google/yamnet/1", metavar="HANDLE",
        help="TensorFlow Hub handle or local SavedModel path for YAMNet when --detector yamnet is used",
    )
    parser.add_argument(
        "--yamnet-score-threshold", type=float, default=0.05, metavar="N",
        help="Segment activation threshold for YAMNet weighted scream/moan scores",
    )
    parser.add_argument(
        "--yamnet-label-debug-json", default="", metavar="FILE",
        help="Optional JSON export with per-window YAMNet scores and top AudioSet labels",
    )
    parser.add_argument(
        "--yamnet-top-k", type=int, default=8, metavar="N",
        help="How many top AudioSet labels to include per window in --yamnet-label-debug-json",
    )
    parser.add_argument(
        "--yamnet-min-window-rms", type=float, default=YAMNET_MIN_WINDOW_RMS, metavar="RMS",
        help="Absolute backstop RMS floor for YAMNet windows before scoring (0 = disable fixed floor)",
    )
    parser.add_argument(
        "--yamnet-context-rms-seconds", type=float, default=YAMNET_CONTEXT_RMS_SECONDS, metavar="SEC",
        help="How much preceding audio context to use for the adaptive YAMNet RMS gate",
    )
    parser.add_argument(
        "--yamnet-context-rms-ratio", type=float, default=YAMNET_CONTEXT_RMS_RATIO, metavar="RATIO",
        help="Adaptive YAMNet gate: require window RMS >= RATIO x preceding-context RMS reference",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=3.0, metavar="N",
        help="Threshold multiplier above whole-file average energy (higher = fewer detections)",
    )
    parser.add_argument(
        "--female-freq", nargs=2, type=float, default=[350.0, 2400.0],
        metavar=("LOW", "HIGH"),
        help="Female vocalization frequency range in Hz",
    )
    parser.add_argument(
        "--male-freq", nargs=2, type=float, default=[100.0, 500.0],
        metavar=("LOW", "HIGH"),
        help="Male vocalization frequency range in Hz",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=11025, metavar="HZ",
        help="Audio sample rate for analysis (Nyquist must exceed female freq high)",
    )
    parser.add_argument(
        "--jobs", type=int, default=min(4, os.cpu_count() or 1), metavar="N",
        help="Number of parallel analysis jobs",
    )
    parser.add_argument(
        "--cache", default=".screamfinder-cache.json", metavar="FILE",
        help="Path to cache file",
    )
    parser.add_argument(
        "--clip-duration", type=_parse_duration, default=0.0, metavar="DUR",
        help="If > 0, analyze only the last DUR of each file (0 = full file). "
             "Accepts seconds (e.g. 300) or [h:]mm:ss (e.g. 5:00 or 1:05:00)",
    )
    parser.add_argument(
        "--min-vocal-duration", type=float, default=0.5, metavar="SEC",
        help="Minimum continuous duration (seconds) to count as a sustained vocalization; "
             "shorter bursts are ignored (0 = count all frames)",
    )
    parser.add_argument(
        "--n-fft", type=int, default=1024, metavar="N",
        help="STFT window size in samples (power of 2; larger = finer frequency resolution but slower)",
    )
    parser.add_argument(
        "--hop-length", type=int, default=1024, metavar="N",
        help="STFT hop size in samples (larger = fewer frames and faster; smaller = finer time resolution)",
    )
    parser.add_argument(
        "--min-audio-rms", type=float, default=0.005, metavar="RMS",
        help="Silence gate: files with whole-file RMS below this are skipped (0 = disable). "
             "Audio is normalised to [-1, 1]; 0.005 ≈ -46 dBFS.",
    )
    parser.add_argument(
        "--noise-floor-pct", type=float, default=10.0, metavar="PCT",
        help="Percentile (0–50) of per-band frame energy used as noise floor reference. "
             "Lower = more sensitive; higher = less sensitive to constant background noise.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable caching",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-analyze all files even if cached",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )

    # Apply config file values as defaults (CLI args still win)
    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    if args.serve_report:
        report_path = Path(args.serve_report)
        try:
            html_text = report_path.read_text(encoding="utf-8")
            media_paths = extract_media_paths_from_report(html_text)
        except Exception as exc:
            print(f"ERROR: could not serve report: {exc}", file=sys.stderr)
            sys.exit(1)
        serve_report(report_path, media_paths, port=args.port, browser=args.browser)
        return

    if not args.paths:
        parser.error("PATH is required unless --serve-report is used")

    # Validate
    args.female_freq = tuple(args.female_freq)  # type: ignore[assignment]
    args.male_freq   = tuple(args.male_freq)     # type: ignore[assignment]

    analysis_sample_rate = YAMNET_SAMPLE_RATE if args.detector == "yamnet" else args.sample_rate
    nyquist = analysis_sample_rate / 2.0
    if args.detector == "heuristic" and args.female_freq[1] > nyquist:
        print(
            f"WARNING: female-freq high ({args.female_freq[1]} Hz) exceeds "
            f"Nyquist ({nyquist} Hz) for sample-rate {analysis_sample_rate}. "
            f"Consider raising --sample-rate.",
            file=sys.stderr,
        )
    if args.detector == "yamnet" and args.sample_rate != YAMNET_SAMPLE_RATE:
        print(
            f"INFO: detector 'yamnet' always decodes audio at {YAMNET_SAMPLE_RATE} Hz; "
            f"ignoring --sample-rate {args.sample_rate}.",
            file=sys.stderr,
        )

    check_dependency("ffmpeg")
    check_dependency("ffprobe")

    # Find files
    media_files = find_media_files(args.paths)
    if not media_files:
        print("No media files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(media_files)} media file(s).", file=sys.stderr)

    # Cache
    cache_path: Optional[Path] = None
    cache: Dict[str, dict] = {}
    if not args.no_cache:
        cache_path = Path(args.cache)
        cache = load_cache(cache_path)

    # Build analysis params dict (also used as the cache key component)
    params: Dict[str, object] = {
        "analysis_cache_version": ANALYSIS_CACHE_VERSION,
        "sample_rate":        args.sample_rate,
        "analysis_sample_rate": analysis_sample_rate,
        "n_fft":              args.n_fft,
        "hop_length":         args.hop_length,
        "female_freq":        list(args.female_freq),
        "male_freq":          list(args.male_freq),
        "threshold":          args.threshold,
        "clip_duration":      args.clip_duration,
        "min_vocal_duration": args.min_vocal_duration,
        "min_audio_rms":      args.min_audio_rms,
        "noise_floor_pct":    args.noise_floor_pct,
        "detector":           args.detector,
        "yamnet_model":       args.yamnet_model,
        "yamnet_score_threshold": args.yamnet_score_threshold,
        "yamnet_min_window_rms": args.yamnet_min_window_rms,
        "yamnet_context_rms_seconds": args.yamnet_context_rms_seconds,
        "yamnet_context_rms_ratio": args.yamnet_context_rms_ratio,
        "yamnet_collect_debug": bool(args.yamnet_label_debug_json),
        "yamnet_top_k":       args.yamnet_top_k,
    }

    # Pre-check cache in the main process; only submit uncached files to workers.
    results: List[dict] = [None] * len(media_files)  # type: ignore[list-item]
    to_run: List[Tuple[int, Path]] = []
    completed = 0
    n_total = len(media_files)

    def _print_progress(name: str, data: Dict[str, object], cached: bool) -> None:
        nonlocal completed
        completed += 1
        metric1_label, metric2_label = detector_metric_labels(str(data.get("detector", args.detector)))
        metric1_name = metric1_label.replace(" %", "").lower()
        tag     = " (cached)" if cached else ""
        dur_str = format_duration(data.get("duration"))  # type: ignore[arg-type]
        fem     = data.get("female_pct", -1.0)
        mal     = data.get("male_pct",   -1.0)
        fem_str = f"{fem:.1f}%" if float(fem) >= 0 else "ERROR"  # type: ignore[arg-type]
        mal_str = f"{mal:.1f}%" if float(mal) >= 0 else "ERROR"  # type: ignore[arg-type]
        metric_txt = f"{metric1_name}={fem_str}"
        if metric2_label is not None:
            metric2_name = metric2_label.replace(" %", "").lower()
            metric_txt += f"  {metric2_name}={mal_str}"
        print(
            f"[{completed}/{n_total}] {name}{tag}: "
            f"dur={dur_str}  {metric_txt}",
            file=sys.stderr,
        )

    for i, vp in enumerate(media_files):
        key = _cache_key(vp, params)
        if not args.force and key in cache:
            data = cache[key]
            results[i] = _make_result(vp, data, cached=True)
            _print_progress(vp.name, data, cached=True)
        else:
            to_run.append((i, vp))

    # Analyze uncached files. For a single worker, stay in-process so local
    # debugging and constrained environments do not need multiprocessing.
    if to_run:
        if args.jobs <= 1:
            for i, vp in to_run:
                try:
                    data = _analyze_worker(str(vp), params)
                except Exception as exc:
                    print(f"ERROR analyzing {vp.name}: {exc}", file=sys.stderr)
                    data = {
                        "duration": 0.0,
                        "female_pct": -1.0,
                        "male_pct": -1.0,
                        "segments": [],
                        "detector": args.detector,
                        "yamnet_debug": None,
                    }
                results[i] = _make_result(vp, data, cached=False)
                key = _cache_key(vp, params)
                cache[key] = data
                if cache_path:
                    save_cache(cache_path, cache)
                _print_progress(vp.name, data, cached=False)
        else:
            # Workers are pure-computation top-level functions; cache I/O stays in main.
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futs = {
                    pool.submit(_analyze_worker, str(vp), params): (i, vp)
                    for i, vp in to_run
                }
                for fut in as_completed(futs):
                    i, vp = futs[fut]
                    try:
                        data = fut.result()
                    except Exception as exc:
                        print(f"ERROR analyzing {vp.name}: {exc}", file=sys.stderr)
                        data = {
                            "duration": 0.0,
                            "female_pct": -1.0,
                            "male_pct": -1.0,
                            "segments": [],
                            "detector": args.detector,
                            "yamnet_debug": None,
                        }
                    results[i] = _make_result(vp, data, cached=False)
                    key = _cache_key(vp, params)
                    cache[key] = data
                    if cache_path:
                        save_cache(cache_path, cache)
                    _print_progress(vp.name, data, cached=False)

    # Generate HTML
    html = generate_html(results, args)
    output_path = Path(args.output)
    output_path.write_text(html, encoding="utf-8")
    abs_path = output_path.resolve()
    print(f"\nReport written to: {abs_path}", file=sys.stderr)
    if args.serve:
        print(
            f"Serving for Safari-compatible playback on http://127.0.0.1:{args.port}/",
            file=sys.stderr,
        )
    else:
        print(f"Open in browser:  file://{abs_path}", file=sys.stderr)
        print(
            "Safari tip: use --serve (or --serve-report) so media outside the "
            "report folder can play.",
            file=sys.stderr,
        )

    if args.segments_json:
        segments_path = Path(args.segments_json)
        export_segments_json(results, segments_path)
        print(f"Segments JSON:    {segments_path.resolve()}", file=sys.stderr)
    if args.yamnet_label_debug_json:
        debug_path = Path(args.yamnet_label_debug_json)
        export_yamnet_debug_json(results, debug_path)
        print(f"YAMNet debug:     {debug_path.resolve()}", file=sys.stderr)

    if args.serve:
        media_paths = [Path(r["path"]) for r in results]
        serve_report(abs_path, media_paths, port=args.port, browser=args.browser)


if __name__ == "__main__":
    main()