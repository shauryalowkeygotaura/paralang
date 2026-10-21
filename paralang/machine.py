"""
machine.py -- is a PERSON on the line? (v0.3 #7) Framework-agnostic.

Outbound calls waste most of their minutes on voicemail, IVRs and hold music
(jio cost levers: "duration > TTS > wasted machine calls"). Classic answering-
machine detection, from signals paralang already has, on the first two turns:

  beep       narrowband tone 600-2000 Hz held >= 0.25 s   -> voicemail (strong)
  words      voicemail / IVR phrases, English + Hindi     -> voicemail / ivr
  greeting   a person answers in 1-3 words then waits; a  -> voicemail (medium)
             recording talks >= 3.5 s straight, 9+ words
  music      >= 4 s of sound, <= 2 words, no pauses        -> music (weak)

Heuristic by design (no model): every verdict carries its evidence, and the
agent is only TOLD (a cue + prompt hint) so it leaves the one-line voicemail
message instead of pitching to a recording.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

BEEP_MIN_S = 0.25
BEEP_BAND_HZ = (600.0, 2000.0)
BEEP_CONCENTRATION = 0.6      # share of a frame's energy within +-1 bin of its peak
GREETING_VOICED_S = 3.5
GREETING_WORDS = 9

_VOICEMAIL = re.compile(
    r"\b(leave (a|your) message|not available|unavailable|after the (tone|beep)|record your message|"
    r"voice ?mail|abhi uplabdh nahi|uplabdh nahin|message chhod|sandesh chhod|vyast hai|"
    r"switched off|band hai|pahunch se bahar|not reachable)\b", re.IGNORECASE)
_IVR = re.compile(
    r"\b(press \d|press (one|two|zero|star|hash)|dial \d|option|please hold|stay on the line|"
    r"your call is important|dabaye|dabayein|dabaiye|line par bane|kripya pratiksha|pratiksha karein|"
    r"aapki call hamare liye)\b", re.IGNORECASE)


@dataclass
class Verdict:
    kind: str                       # human | voicemail | ivr | music
    confidence: float
    evidence: dict = field(default_factory=dict)


def beep_seconds(x: np.ndarray, sr: int, frame_s: float = 0.02) -> float:
    """Longest run of narrowband 600-2000 Hz tone frames (a voicemail beep).
    Speech never concentrates 60% of a frame's energy in one FFT bin; a beep does."""
    n = max(16, int(sr * frame_s))
    frames = x[: len(x) // n * n].reshape(-1, n)
    if frames.size == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2
    total = spec.sum(axis=1) + 1e-12
    peak = spec.argmax(axis=1)
    rows = np.arange(len(spec))
    lo_i, hi_i = np.clip(peak - 1, 0, None), np.clip(peak + 1, None, spec.shape[1] - 1)
    conc = (spec[rows, lo_i] + spec[rows, peak] + spec[rows, hi_i]) / total
    freq = peak * sr / n
    loud = 10 * np.log10(total) > 10 * np.log10(np.percentile(total, 95)) - 25
    tone = (conc >= BEEP_CONCENTRATION) & (freq >= BEEP_BAND_HZ[0]) & (freq <= BEEP_BAND_HZ[1]) & loud
    best = run = 0
    for v in tone:
        run = run + 1 if v else 0
        best = max(best, run)
    return best * frame_s


def classify(features, text: str, turn: int) -> Verdict:
    """Verdict for caller turn `turn` (1-based). Only turns 1-2 are judged."""
    if turn > 2:
        return Verdict("human", 0.0)
    beep = float(features.extras.get("beep_s", 0.0) or 0.0)
    if beep >= BEEP_MIN_S:
        return Verdict("voicemail", 0.9, {"beep_s": beep})
    if _IVR.search(text or ""):
        return Verdict("ivr", 0.85, {"phrase": _IVR.search(text).group(0)})
    if _VOICEMAIL.search(text or ""):
        return Verdict("voicemail", 0.85, {"phrase": _VOICEMAIL.search(text).group(0)})
    if turn == 1 and features.voiced_s >= GREETING_VOICED_S and features.words >= GREETING_WORDS:
        return Verdict("voicemail", 0.7, {"voiced_s": features.voiced_s, "words": features.words})
    if features.voiced_s >= 4.0 and features.words <= 2 and features.pause_count == 0:
        return Verdict("music", 0.6, {"voiced_s": features.voiced_s, "words": features.words})
    return Verdict("human", 0.0)
