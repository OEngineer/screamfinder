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
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def media_kind(path: Path) -> str:
    """Return "audio" if path's suffix is a known audio extension, else "video"."""
    return "audio" if path.suffix.lower() in AUDIO_EXTENSIONS else "video"

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
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: rgba(20,20,30,0.95);
  border-bottom: 1px solid #222;
  flex-shrink: 0;
}

.player-title {
  flex: 1;
  font-size: 13px;
  color: #bbb;
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
   give the player a sensible minimum height so the cover has room to render. */
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

.progress-buffered {
  position: absolute;
  top: 0; left: 0; height: 100%;
  background: rgba(255,255,255,0.25);
  border-radius: 3px;
  pointer-events: none;
}

.progress-fill {
  position: absolute;
  top: 0; left: 0; height: 100%;
  background: var(--pinkf);
  border-radius: 3px;
  pointer-events: none;
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
}
.progress-track:hover .progress-thumb { transform: translate(-50%, -50%) scale(1); }

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

/* Cursor hiding when controls hidden and playing */
.player-wrap.hide-cursor { cursor: none; }
"""

# ---------------------------------------------------------------------------
# Embedded JavaScript
# ---------------------------------------------------------------------------

JS = r"""
// ── Data injected above ───────────────────────────────────────────────────

const maxDuration = Math.max(...DATA.map(d => d.duration), 1);

// ── Scoring ───────────────────────────────────────────────────────────────

function getWeights() {
  return {
    dur: parseFloat(document.getElementById('w-dur').value),
    fem: parseFloat(document.getElementById('w-fem').value),
    mal: parseFloat(document.getElementById('w-mal').value),
  };
}

function computeScore(item, w) {
  return w.dur * (item.duration / maxDuration)
       + w.fem * (item.female_pct / 100)
       + w.mal * (item.male_pct  / 100);
}

// ── Table rendering ───────────────────────────────────────────────────────

let sortCol = 'score';
let sortAsc  = false;

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
  tbody.innerHTML = rows.map((item, rank) => {
    const scoreNorm = (item.score / maxScore) * 100;
    const femW = Math.min(item.female_pct, 100);
    const malW = Math.min(item.male_pct,  100);
    const femTxt = item.female_pct >= 0 ? item.female_pct.toFixed(1) + '%' : '—';
    const malTxt = item.male_pct   >= 0 ? item.male_pct.toFixed(1)   + '%' : '—';

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
      <td class="bar-cell bar-m">
        <div class="bar-fill" style="width:${malW}%"></div>
        <span class="bar-text">${malTxt}</span>
      </td>
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
}

// Sliders
['w-dur','w-fem','w-mal'].forEach(id => {
  const sl = document.getElementById(id);
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
const playerError     = document.getElementById('player-error');
const playerErrDetail = document.getElementById('player-error-detail');
const playerErrPath   = document.getElementById('player-error-path');
const audioCover      = document.getElementById('audio-cover');
const audioCoverName  = document.getElementById('audio-cover-name');
const progressTrack  = document.getElementById('progress-track');
const progressFill   = document.getElementById('progress-fill');
const progressBuf    = document.getElementById('progress-buf');
const progressThumb  = document.getElementById('progress-thumb');
const timeCurEl      = document.getElementById('time-cur');
const timeTotEl      = document.getElementById('time-tot');
const playBtn        = document.getElementById('btn-play');
const muteBtn        = document.getElementById('btn-mute');
const volSlider      = document.getElementById('vol-track');

let isDragging    = false;
let savedVol      = 1;
let hideCtrlTimer = null;

// Open / close
function openPlayer(idx) {
  const item = DATA[idx];
  const isAudio = item.kind === 'audio';
  document.getElementById('player-title').textContent = item.name;
  playerError.classList.add('hidden');
  playerControls.style.display = '';
  playerWrap.classList.toggle('is-audio', isAudio);
  audioCover.classList.toggle('hidden', !isAudio);
  audioCoverName.textContent = isAudio ? item.name : '';
  video.src = item.url;
  modal.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
  showControls();
  video.play().catch(() => {});
}

video.addEventListener('error', () => {
  const e = video.error;
  const CODE = {1:'ABORTED', 2:'NETWORK', 3:'DECODE', 4:'SRC_NOT_SUPPORTED'};
  const detail = e ? `${CODE[e.code] || 'ERR'} (code ${e.code})${e.message ? ': ' + e.message : ''}` : 'Unknown error';
  // Convert file:// URL back to a plain path for display
  const rawPath = decodeURIComponent(video.src.replace(/^file:\/\//, ''));
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
  clearTimeout(hideCtrlTimer);
}

document.getElementById('modal-backdrop').addEventListener('click', closePlayer);

// Controls visibility
function showControls() {
  playerControls.classList.remove('hidden-ctrl');
  playerWrap.classList.remove('hide-cursor');
  clearTimeout(hideCtrlTimer);
  if (!video.paused) {
    hideCtrlTimer = setTimeout(() => {
      playerControls.classList.add('hidden-ctrl');
      playerWrap.classList.add('hide-cursor');
    }, 3000);
  }
}

playerWrap.addEventListener('mousemove', showControls);
video.addEventListener('play',  () => {
  playBtn.textContent = '⏸';
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
video.addEventListener('click',         () => { togglePlay(); showControls(); });
video.addEventListener('dblclick',      reqFullscreen);

// Mouse wheel → volume
video.addEventListener('wheel', e => {
  e.preventDefault();
  video.muted  = false;
  video.volume = Math.max(0, Math.min(1, video.volume + (e.deltaY < 0 ? 0.05 : -0.05)));
  updateMuteIcon();
  showControls();
}, { passive: false });

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
  <<<FILE_COUNT>>> file(s) &bull;
  Female: <<<FFREQ_LO>>>–<<<FFREQ_HI>>> Hz &bull;
  Male: <<<MFREQ_LO>>>–<<<MFREQ_HI>>> Hz &bull;
  Threshold: <<<THRESHOLD>>>× &bull;
  Noise floor: <<<NOISE_FLOOR_PCT>>>th pct &bull;
  Min sustained: <<<MIN_VOC_DUR>>>s<<<CLIP_INFO>>>
</p>

<div class="controls">
  <div class="controls-title">Sort Weights</div>
  <div class="sliders">
    <div class="slider-group">
      <label><span class="swatch" style="background:#7878a0"></span>Duration</label>
      <input type="range" id="w-dur" min="0" max="5" step="0.1" value="1.0">
      <span class="slider-val" id="w-dur-val">1.0</span>
    </div>
    <div class="slider-group">
      <label><span class="swatch" style="background:#e91e8c"></span>Female %</label>
      <input type="range" id="w-fem" min="0" max="5" step="0.1" value="2.0">
      <span class="slider-val" id="w-fem-val">2.0</span>
    </div>
    <div class="slider-group">
      <label><span class="swatch" style="background:#2196f3"></span>Male %</label>
      <input type="range" id="w-mal" min="0" max="5" step="0.1" value="1.0">
      <span class="slider-val" id="w-mal-val">1.0</span>
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
        <th data-col="female_pct">Female %</th>
        <th data-col="male_pct">Male %</th>
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
      <span id="player-title" class="player-title"></span>
      <button class="hdr-btn" onclick="closePlayer()" title="Close (Esc)">✕</button>
    </div>
    <video id="player" preload="metadata"></video>
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
        Browsers only support MP4/H.264, WebM, and Ogg natively.
      </div>
    </div>
    <div id="player-controls" class="player-controls">
      <div class="progress-area">
        <span id="time-cur" class="time-txt">0:00</span>
        <div id="progress-track" class="progress-track">
          <div id="progress-buf"   class="progress-buffered" style="width:0%"></div>
          <div id="progress-fill"  class="progress-fill"     style="width:0%"></div>
          <div id="progress-thumb" class="progress-thumb"    style="left:0%"></div>
        </div>
        <span id="time-tot" class="time-txt" style="text-align:right">0:00</span>
      </div>
      <div class="ctrl-row">
        <button id="btn-play" class="ctrl-btn" onclick="togglePlay()" title="Play/Pause (Space)">▶</button>
        <button class="ctrl-btn" onclick="seek(-30)" title="Back 30s (Shift+←)">⏮ 30s</button>
        <button class="ctrl-btn" onclick="seek(-5)"  title="Back 5s (←)">⏪ 5s</button>
        <button class="ctrl-btn" onclick="seek(5)"   title="Fwd 5s (→)">5s ⏩</button>
        <button class="ctrl-btn" onclick="seek(30)"  title="Fwd 30s (Shift+→)">30s ⏭</button>
        <span class="spacer"></span>
        <button id="btn-mute" class="ctrl-btn" onclick="toggleMute()" title="Mute (M)">🔊</button>
        <input type="range" id="vol-track" class="vol-slider" min="0" max="1" step="0.02" value="1">
        <button class="ctrl-btn" onclick="reqFullscreen()" title="Fullscreen (F)">⛶</button>
      </div>
      <div style="margin-top:6px">
        <span class="kbd-hint">Keys: Space play/pause &bull; ←→ ±5s &bull; Shift+←→ ±30s &bull; ↑↓ volume &bull; M mute &bull; F fullscreen &bull; 0–9 jump &bull; Esc close</span>
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
    sample_rate     = int(params["sample_rate"])      # type: ignore[arg-type]
    n_fft           = int(params["n_fft"])            # type: ignore[arg-type]
    hop_length      = int(params["hop_length"])       # type: ignore[arg-type]
    female_freq     = tuple(params["female_freq"])    # type: ignore[arg-type]
    male_freq       = tuple(params["male_freq"])      # type: ignore[arg-type]
    threshold       = float(params["threshold"])      # type: ignore[arg-type]
    min_voc_dur     = float(params["min_vocal_duration"])  # type: ignore[arg-type]
    min_audio_rms   = float(params["min_audio_rms"]) # type: ignore[arg-type]
    noise_floor_pct = float(params["noise_floor_pct"])  # type: ignore[arg-type]
    clip_duration   = float(params["clip_duration"]) # type: ignore[arg-type]

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

    audio = extract_audio(video_path, sample_rate, start_sec=start_sec, duration_sec=extract_dur)
    if audio is None:
        # Audio extraction failed; fall back to ffprobe for duration.
        duration = reported_duration or get_media_duration(video_path) or 0.0
        return {"duration": duration, "female_pct": -1.0, "male_pct": -1.0}

    # Use ffprobe-reported full duration when clipping; otherwise derive from audio.
    duration = reported_duration if reported_duration is not None else len(audio) / sample_rate
    female_pct, male_pct = analyze_vocalizations(
        audio, sample_rate, n_fft, hop_length,
        female_freq, male_freq, threshold, min_voc_dur,  # type: ignore[arg-type]
        min_audio_rms, noise_floor_pct,
    )
    return {"duration": duration, "female_pct": female_pct, "male_pct": male_pct}


def _make_result(video_path: Path, data: Dict[str, object], cached: bool) -> dict:
    """Build the per-file result dict from worker output."""
    return {
        "path":       str(video_path),
        "url":        video_path.as_uri(),
        "name":       video_path.name,
        "kind":       media_kind(video_path),
        "duration":   data.get("duration"),
        "female_pct": data.get("female_pct", -1.0),
        "male_pct":   data.get("male_pct",   -1.0),
        "cached":     cached,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(results: List[dict], args: argparse.Namespace) -> str:
    max_duration = max((r["duration"] or 0 for r in results), default=1.0) or 1.0

    js_data = []
    for r in results:
        dur = r["duration"] or 0
        js_data.append({
            "name":         r["name"],
            "url":          r["url"],
            "kind":         r.get("kind", "video"),
            "duration":     round(dur, 3),
            "duration_fmt": format_duration(r["duration"]),
            "female_pct":   round(max(r["female_pct"], 0), 2),
            "male_pct":     round(max(r["male_pct"],   0), 2),
        })

    data_json = json.dumps(js_data, ensure_ascii=True)

    html = (
        HTML_TEMPLATE
        .replace("<<<CSS>>>",       CSS)
        .replace("<<<JS>>>",        JS)
        .replace("<<<DATA_JSON>>>", data_json)
        .replace("<<<FILE_COUNT>>>", str(len(results)))
        .replace("<<<FFREQ_LO>>>",  str(args.female_freq[0]))
        .replace("<<<FFREQ_HI>>>",  str(args.female_freq[1]))
        .replace("<<<MFREQ_LO>>>",  str(args.male_freq[0]))
        .replace("<<<MFREQ_HI>>>",  str(args.male_freq[1]))
        .replace("<<<THRESHOLD>>>",       str(args.threshold))
        .replace("<<<NOISE_FLOOR_PCT>>>", str(args.noise_floor_pct))
        .replace("<<<MIN_VOC_DUR>>>",     str(args.min_vocal_duration))
        .replace("<<<CLIP_INFO>>>", (
            f" &bull; Clip: last {args.clip_duration:g}s"
            if args.clip_duration > 0 else ""
        ))
    )
    return html


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
    str_keys   = ("output", "cache")
    float_keys = ("threshold", "clip_duration", "min_vocal_duration", "min_audio_rms", "noise_floor_pct")
    int_keys   = ("sample_rate", "jobs", "n_fft", "hop_length")
    bool_keys  = ("no_cache", "force")

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
        "paths", nargs="+", metavar="PATH",
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

    # Validate
    args.female_freq = tuple(args.female_freq)  # type: ignore[assignment]
    args.male_freq   = tuple(args.male_freq)     # type: ignore[assignment]

    nyquist = args.sample_rate / 2.0
    if args.female_freq[1] > nyquist:
        print(
            f"WARNING: female-freq high ({args.female_freq[1]} Hz) exceeds "
            f"Nyquist ({nyquist} Hz) for sample-rate {args.sample_rate}. "
            f"Consider raising --sample-rate.",
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
        "sample_rate":        args.sample_rate,
        "n_fft":              args.n_fft,
        "hop_length":         args.hop_length,
        "female_freq":        list(args.female_freq),
        "male_freq":          list(args.male_freq),
        "threshold":          args.threshold,
        "clip_duration":      args.clip_duration,
        "min_vocal_duration": args.min_vocal_duration,
        "min_audio_rms":      args.min_audio_rms,
        "noise_floor_pct":    args.noise_floor_pct,
    }

    # Pre-check cache in the main process; only submit uncached files to workers.
    results: List[dict] = [None] * len(media_files)  # type: ignore[list-item]
    to_run: List[Tuple[int, Path]] = []
    completed = 0
    n_total = len(media_files)

    def _print_progress(name: str, data: Dict[str, object], cached: bool) -> None:
        nonlocal completed
        completed += 1
        tag     = " (cached)" if cached else ""
        dur_str = format_duration(data.get("duration"))  # type: ignore[arg-type]
        fem     = data.get("female_pct", -1.0)
        mal     = data.get("male_pct",   -1.0)
        fem_str = f"{fem:.1f}%" if float(fem) >= 0 else "ERROR"  # type: ignore[arg-type]
        mal_str = f"{mal:.1f}%" if float(mal) >= 0 else "ERROR"  # type: ignore[arg-type]
        print(
            f"[{completed}/{n_total}] {name}{tag}: "
            f"dur={dur_str}  female={fem_str}  male={mal_str}",
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

    # Analyze uncached files with ProcessPoolExecutor (true CPU parallelism).
    # Workers are pure-computation top-level functions; cache I/O stays in main.
    if to_run:
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
                    data = {"duration": 0.0, "female_pct": -1.0, "male_pct": -1.0}
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
    print(f"Open in browser:  file://{abs_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
