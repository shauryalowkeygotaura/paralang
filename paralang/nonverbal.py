"""
nonverbal.py -- laughs, sighs, breaths: sounds Whisper drops (v0.3 #8).

HEURISTIC, and deliberately conservative: a missed laugh costs nothing, a false
"sigh" would wrongly mark a lead as reluctant. Every event needs either an
acoustic pattern that ordinary speech does not make, or the transcript tokens
Whisper sometimes writes for it ("haha", "(laughs)", "uff").

  laugh   >= 3 energy bursts at 3.5-7 per second with REGULAR spacing (interval
          CV < 0.25) and breathy gaps between them. Speech syllables have the
          same rate but irregular spacing; that regularity is the tell.
  sigh    >= 0.45 s mostly UNvoiced (no pitch) sound, clearly above the noise
          floor but well below speech level, with falling energy: an exhale.
  breath  0.15-0.4 s unvoiced burst between speech; counted only.

Frames: 20 ms energy (features._frame_db) and the 10 ms pitch track (0 = unvoiced).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

FRAME_S = 0.02
LAUGH_RATE_HZ = (3.5, 7.0)
LAUGH_MIN_BURSTS = 3
LAUGH_MAX_CV = 0.25
SIGH_MIN_S = 0.45

_LAUGH_WORDS = re.compile(r"(\b(ha){2,}h?\b|\bhehe+\b|\(laugh\w*\)|\[laugh\w*\]|हा\s?हा|\blol\b)", re.I)
_SIGH_WORDS = re.compile(r"(\buff+\b|\bhaa+h\b|\(sigh\w*\)|\[sigh\w*\]|\bhaaye\b|उफ़?)", re.I)


@dataclass
class Event:
    kind: str           # laugh | sigh | breath
    start_s: float
    end_s: float
    strength: float
    via: str            # acoustic | words | both

    def to_dict(self) -> dict:
        return {"kind": self.kind, "start_s": round(self.start_s, 2), "end_s": round(self.end_s, 2),
                "strength": round(self.strength, 2), "via": self.via}


def _voiced_per_energy_frame(hz: np.ndarray, n_frames: int) -> np.ndarray:
    """Map the 10 ms pitch track onto 20 ms energy frames (voiced if either half is)."""
    v = hz > 0
    v = v[: (len(v) // 2) * 2].reshape(-1, 2).any(axis=1) if len(v) >= 2 else v
    out = np.zeros(n_frames, dtype=bool)
    out[: min(n_frames, len(v))] = v[: n_frames]
    return out


def _bursts(rms_db: np.ndarray, floor: float, peak: float) -> list[int]:
    """Indices of local energy peaks near SPEAKING level (candidate syllables).
    Must be within 10 dB of the turn's peak: the breathy gaps between laugh
    bursts have their own little noise maxima and must not count as bursts."""
    thr = max(floor + 8.0, peak - 10.0)
    peaks = []
    for i in range(1, len(rms_db) - 1):
        if rms_db[i] > thr and rms_db[i] >= rms_db[i - 1] and rms_db[i] > rms_db[i + 1]:
            if not peaks or i - peaks[-1] >= 3:          # >= 60 ms apart
                peaks.append(i)
    return peaks


def _laugh_acoustic(rms_db: np.ndarray, voiced: np.ndarray, floor: float, peak: float) -> Event | None:
    peaks = _bursts(rms_db, floor, peak)
    best = None
    i = 0
    while i < len(peaks) - LAUGH_MIN_BURSTS + 1:
        j = i + 1
        while j < len(peaks):
            iv = np.diff(peaks[i:j + 1]) * FRAME_S
            rate = 1.0 / iv.mean()
            cv = iv.std() / iv.mean()
            if not (LAUGH_RATE_HZ[0] <= rate <= LAUGH_RATE_HZ[1] and cv <= LAUGH_MAX_CV):
                break
            j += 1
        run = peaks[i:j]
        if len(run) >= LAUGH_MIN_BURSTS:
            # breathy gaps: the troughs between bursts are mostly unvoiced
            gaps = np.concatenate([np.arange(a + 1, b) for a, b in zip(run, run[1:])]) if len(run) > 1 else []
            breathy = 1.0 - (voiced[gaps].mean() if len(gaps) else 1.0)
            if breathy >= 0.5:
                s = min(1.0, 0.5 + 0.1 * (len(run) - LAUGH_MIN_BURSTS) + 0.3 * breathy)
                ev = Event("laugh", run[0] * FRAME_S, run[-1] * FRAME_S, s, "acoustic")
                best = ev if best is None or ev.strength > best.strength else best
            i = j - 1
        else:
            i += 1
    return best


def _sigh_acoustic(rms_db: np.ndarray, voiced: np.ndarray, floor: float, peak: float) -> Event | None:
    """Longest run of unvoiced, mid-level frames with falling energy."""
    mid = (rms_db > floor + 6.0) & (rms_db < peak - 6.0) & ~voiced
    best, start = None, None
    for i, m in enumerate(np.append(mid, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            n = i - start
            if n * FRAME_S >= SIGH_MIN_S:
                seg = rms_db[start:i]
                slope = np.polyfit(np.arange(n) * FRAME_S, seg, 1)[0]   # dB per second
                if slope < -4.0:                                          # it fades out
                    s = min(1.0, 0.4 + 0.3 * (n * FRAME_S - SIGH_MIN_S) / 0.5 + 0.02 * -slope)
                    ev = Event("sigh", start * FRAME_S, i * FRAME_S, s, "acoustic")
                    best = ev if best is None or ev.strength > best.strength else best
            start = None
    return best


def _breaths(rms_db: np.ndarray, voiced: np.ndarray, floor: float, peak: float) -> int:
    mid = (rms_db > floor + 4.0) & (rms_db < peak - 10.0) & ~voiced
    count, run = 0, 0
    for m in np.append(mid, False):
        if m:
            run += 1
        else:
            if 0.15 <= run * FRAME_S <= 0.4:
                count += 1
            run = 0
    return count


def detect(rms_db: np.ndarray, hz: np.ndarray, text: str = "") -> tuple[list[Event], int]:
    """(laugh/sigh events, breath count) for one caller turn."""
    if rms_db.size < 20:
        return [], 0
    floor = float(np.percentile(rms_db, 10))
    peak = float(np.percentile(rms_db, 95))
    if peak - floor < 10:          # no level contrast: cannot separate a sigh from speech
        return [], 0
    voiced = _voiced_per_energy_frame(hz, rms_db.size)
    events: list[Event] = []
    laugh = _laugh_acoustic(rms_db, voiced, floor, peak)
    if _LAUGH_WORDS.search(text or ""):
        laugh = (Event("laugh", laugh.start_s, laugh.end_s, min(1.0, laugh.strength + 0.3), "both")
                 if laugh else Event("laugh", 0.0, 0.0, 0.7, "words"))
    if laugh:
        events.append(laugh)
    sigh = _sigh_acoustic(rms_db, voiced, floor, peak)
    if _SIGH_WORDS.search(text or ""):
        sigh = (Event("sigh", sigh.start_s, sigh.end_s, min(1.0, sigh.strength + 0.3), "both")
                if sigh else Event("sigh", 0.0, 0.0, 0.6, "words"))
    if sigh:
        events.append(sigh)
    return events, _breaths(rms_db, voiced, floor, peak)


def merge_words(events: list[dict], text: str) -> list[dict]:
    """Add/boost events from the transcript once it exists. Live adapters measure
    audio BEFORE STT returns, so the word evidence arrives separately."""
    out = [dict(e) for e in events]
    for kind, rx, base in (("laugh", _LAUGH_WORDS, 0.7), ("sigh", _SIGH_WORDS, 0.6)):
        if not rx.search(text or ""):
            continue
        hit = next((e for e in out if e["kind"] == kind), None)
        if hit is None:
            out.append(Event(kind, 0.0, 0.0, base, "words").to_dict())
        elif hit["via"] == "acoustic":
            hit["strength"], hit["via"] = round(min(1.0, hit["strength"] + 0.3), 2), "both"
    return out
