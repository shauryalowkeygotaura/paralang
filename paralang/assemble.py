"""
assemble.py -- stitch per-sentence TTS clips into ONE take that sounds recorded.

Every engine paralang drives (Aura, Orpheus, edge) is called once per sentence
or chunk, and naive concatenation leaves four "this is a machine" tells that no
emotion tag can fix:

  1. Padding. Each clip carries the engine's own lead-in / tail silence
     (measured 80-400 ms, different per engine and per clip), so the planner's
     gaps came out longer and uneven. trim() cuts each clip to its speech.
  2. Dead air. Gaps were digital zero. A real recording never is: the ear
     hears the floor vanish and reads it as splicing. Gaps get quiet room tone.
  3. Loudness jumps. Separate requests come back at different levels, so
     sentences sounded like different takes. Clips are matched on SPEECH RMS
     (silence excluded), then the line's own emotion volume is applied on top,
     so "whisper" is still quieter than "excited" -- on Aura too, which has no
     volume knob of its own. A gain change never retimes audio, so this is
     safe for Aura (the never-atempo rule is about time-stretching).
  4. Clicks. Hard cuts at clip edges; every edge gets a short fade.

Numpy only. PCM in and out: 24 kHz mono s16le, like the rest of paralang.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 24000
FRAME_S = 0.01
SPEECH_DB_BELOW_PEAK = 40.0   # a frame is speech if within this of the clip's loud frames
SPEECH_FLOOR_DBFS = -55.0     # ... and above this absolute floor
LEAD_KEEP_S = 0.03            # keep a little before the first speech frame (onsets, "h", "s")
TAIL_KEEP_S = 0.08            # keep more after the last: word endings decay slowly
FADE_S = 0.008
ROOM_TONE_DBFS = -62.0
TARGET_SPEECH_DBFS = -20.0
MAX_GAIN_DB = 9.0             # never boost/cut a clip further than this
PEAK_CEILING = 0.97


@dataclass
class Clip:
    pcm: bytes                 # s16le mono at SR
    pause_before_ms: int = 0   # silence before this clip (planner's gap)
    gain_db: float = 0.0       # emotion volume offset, applied after loudness matching
    trim: bool = True          # False: keep the clip's own edges (e.g. already one take)


def _f32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768.0


def _frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(sr * FRAME_S))
    frames = x[: len(x) // n * n].reshape(-1, n)
    if frames.size == 0:
        return np.zeros(0)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def _speech_mask(x: np.ndarray, sr: int) -> np.ndarray:
    db = _frame_db(x, sr)
    if db.size == 0:
        return np.zeros(0, dtype=bool)
    peak = np.percentile(db, 95)
    return (db > peak - SPEECH_DB_BELOW_PEAK) & (db > SPEECH_FLOOR_DBFS)


def trim(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Cut leading/trailing silence, keeping a short margin each side."""
    mask = _speech_mask(x, sr)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return x
    n = int(sr * FRAME_S)
    start = max(0, idx[0] * n - int(sr * LEAD_KEEP_S))
    end = min(len(x), (idx[-1] + 1) * n + int(sr * TAIL_KEEP_S))
    return x[start:end]


def speech_dbfs(x: np.ndarray, sr: int = SR) -> float | None:
    """RMS level of the speech frames only (pauses would drag a plain RMS down)."""
    mask = _speech_mask(x, sr)
    if not mask.any():
        return None
    n = int(sr * FRAME_S)
    frames = x[: len(mask) * n].reshape(-1, n)[mask]
    return float(20 * np.log10(np.sqrt((frames ** 2).mean()) + 1e-9))


def room_tone(n: int, seed: int = 0, dbfs: float = ROOM_TONE_DBFS) -> np.ndarray:
    """n samples of soft pink-ish noise (deterministic: same seed, same tone)."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    # white noise shaped to 1/f in the frequency domain (pink, no hiss), with
    # the top end rolled off: sounds like a quiet room, not a fan
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec *= 1 / np.sqrt(np.maximum(f, 20.0)) / (1 + (f / 6000) ** 2)
    pink = np.fft.irfft(spec, n).astype(np.float32)
    rms = float(np.sqrt((pink ** 2).mean())) or 1.0
    return pink * (10 ** (dbfs / 20) / rms)


_TONE_CACHE: dict[int, np.ndarray] = {}


def _tone(n: int) -> np.ndarray:
    """Room tone from a cached 4 s loop (generating per gap was slow in pure Python)."""
    base = _TONE_CACHE.get(SR)
    if base is None:
        base = _TONE_CACHE[SR] = room_tone(SR * 4)
    if n <= len(base):
        return base[:n].copy()
    return np.tile(base, n // len(base) + 1)[:n]


def _fade(x: np.ndarray, sr: int) -> np.ndarray:
    k = min(len(x) // 2, int(sr * FADE_S))
    if k > 1:
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
        x = x.copy()
        x[:k] *= ramp
        x[-k:] *= ramp[::-1]
    return x


def join(clips: list[Clip], sr: int = SR, *, tone: bool = True, level: bool = True) -> bytes:
    """Clips -> one PCM take: trimmed, level-matched, faded, room tone in the gaps."""
    parts: list[np.ndarray] = []
    for i, c in enumerate(clips):
        x = _f32(c.pcm)
        if c.trim:
            x = trim(x, sr)
        if level:
            lvl = speech_dbfs(x, sr)
            if lvl is not None:
                g = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, TARGET_SPEECH_DBFS - lvl))
                x = x * 10 ** ((g + c.gain_db) / 20)
        x = _fade(x, sr)
        gap = int(sr * c.pause_before_ms / 1000) if i else 0
        if gap:
            parts.append(_tone(gap) if tone else np.zeros(gap, dtype=np.float32))
        if tone and len(x):
            # tone under speech too, so the floor never switches on and off at clip edges
            x = x + _tone(len(x))
        parts.append(x)
    if not parts:
        return b""
    y = np.concatenate(parts)
    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak > PEAK_CEILING:
        y = y * (PEAK_CEILING / peak)
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()


def raw(clips: list[Clip], sr: int = SR) -> bytes:
    """The old behaviour: plain concatenation with zero-filled gaps."""
    out = bytearray()
    for i, c in enumerate(clips):
        if i:
            out += b"\x00\x00" * (sr * c.pause_before_ms // 1000)
        out += c.pcm
    return bytes(out)


def volume_db(volume_pct: int) -> float:
    """Planner volume % (edge semantics) -> dB gain."""
    return float(20 * np.log10(max(0.05, 1 + volume_pct / 100)))
