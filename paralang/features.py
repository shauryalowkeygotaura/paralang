"""
features.py -- one caller turn of audio (+ its transcript) -> TurnFeatures.

Everything here is *measurement*: numbers you could check with a stopwatch or
a tuner. Interpretation (is this caller hesitant?) lives in cues.py, so the
features stay objective and the judgement stays in one small, tunable place.

Two sources:
  - numpy on 20 ms frames: energy, voiced regions, pauses inside the turn.
    Cheap and robust on 8 kHz phone audio.
  - openSMILE eGeMAPS low-level descriptors (10 ms hop): pitch in semitones,
    loudness, jitter, shimmer, HNR. The standard research feature set, so the
    numbers mean the same thing as in the papers.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

import numpy as np

log = logging.getLogger("paralang")

FRAME_S = 0.02            # energy frame for pause detection
MIN_PAUSE_S = 0.25        # a gap shorter than this is articulation, not a pause
MIN_ACOUSTIC_S = 0.4      # below this much voiced audio, pitch stats are noise
FINAL_WINDOW_S = 0.4      # "how did the turn end": last 400 ms of voiced pitch
SILENCE_DB_BELOW_PEAK = 30.0  # frame is silent if this far below the turn's loud frames

# Hesitation fillers in English + Hinglish. Whole-word, case-insensitive.
# Deliberately NOT included: "accha", "haan", "theek" (those are answers).
FILLERS = ("um", "umm", "uh", "uhh", "hmm", "hm", "er", "erm", "matlab", "woh", "wo",
           "actually", "basically", "like")
_FILLER_RE = re.compile(r"\b(" + "|".join(FILLERS) + r")\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass
class TurnFeatures:
    duration_s: float
    voiced_s: float
    words: int
    words_per_sec: float | None          # words / voiced time (articulation rate)
    pause_count: int
    pause_total_s: float
    longest_pause_s: float
    pause_ratio: float                   # pause_total / (first voice .. last voice)
    fillers: int
    response_latency_s: float | None = None   # agent stopped -> caller started (adapter-supplied)
    barge_in: bool = False                    # caller started while agent was talking
    loudness: float | None = None             # eGeMAPS loudness, voiced mean
    f0_median_st: float | None = None         # semitones re 27.5 Hz (12 st = 1 octave)
    f0_range_st: float | None = None          # p90 - p10, pitch liveliness
    f0_final_slope_st: float | None = None    # st/s over last 400 ms voiced; >0 = rising end
    jitter: float | None = None
    shimmer_db: float | None = None
    hnr_db: float | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def to_float_mono(audio, sample_rate: int, num_channels: int = 1) -> np.ndarray:
    """Accept int16 PCM bytes, int16 ndarray or float ndarray; return float32 mono in [-1, 1]."""
    if sample_rate <= 0 or num_channels <= 0:
        raise ValueError(f"bad audio format: sample_rate={sample_rate} num_channels={num_channels}")
    if isinstance(audio, (bytes, bytearray, memoryview)):
        x = np.frombuffer(bytes(audio), dtype=np.int16).astype(np.float32) / 32768.0
    else:
        x = np.asarray(audio)
        x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
    if num_channels > 1:
        x = x[: len(x) - len(x) % num_channels].reshape(-1, num_channels).mean(axis=1)
    return x


def _voiced_mask(x: np.ndarray, sr: int) -> np.ndarray:
    """Per-20ms-frame speech mask, relative to this turn's own loud frames.

    Relative (not absolute dB) so a quiet phone line and a loud headset get the
    same treatment. The 95th-percentile frame stands in for "speaking level".
    """
    n = max(1, int(sr * FRAME_S))
    frames = x[: len(x) // n * n].reshape(-1, n)
    if frames.size == 0:
        return np.zeros(0, dtype=bool)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    peak = np.percentile(rms_db, 95)
    return (rms_db > peak - SILENCE_DB_BELOW_PEAK) & (rms_db > -60)


def _pauses(mask: np.ndarray) -> tuple[list[float], float]:
    """Silent runs strictly between the first and last voiced frame, plus that span."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return [], 0.0
    inner = mask[idx[0]: idx[-1] + 1]
    pauses, run = [], 0
    for v in inner:
        if v:
            if run * FRAME_S >= MIN_PAUSE_S:
                pauses.append(run * FRAME_S)
            run = 0
        else:
            run += 1
    return pauses, len(inner) * FRAME_S


_SMILE = None


def _smile():
    global _SMILE
    if _SMILE is None:
        import opensmile  # lazy: keeps import of paralang cheap, and opensmile optional
        _SMILE = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                 feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    return _SMILE


def _acoustics(x: np.ndarray, sr: int) -> dict:
    """Pitch/voice-quality stats. Any failure degrades to {} (timing features
    still ship) because one bad turn must never break a live call."""
    try:
        return _acoustics_unchecked(x, sr)
    except Exception as e:  # noqa: BLE001 - boundary with a native lib
        log.warning("paralang: acoustic features skipped (%s: %s)", type(e).__name__, e)
        return {}


def _acoustics_unchecked(x: np.ndarray, sr: int) -> dict:
    lld = _smile().process_signal(x, sr)
    f0 = lld["F0semitoneFrom27.5Hz_sma3nz"].to_numpy()
    voiced = f0 > 0                                  # eGeMAPS writes 0 where unvoiced
    if voiced.sum() < 5:
        return {}
    hop = 0.01
    t = np.arange(len(f0)) * hop
    vf0, vt = f0[voiced], t[voiced]
    tail = vt >= vt[-1] - FINAL_WINDOW_S
    slope = float(np.polyfit(vt[tail], vf0[tail], 1)[0]) if tail.sum() >= 5 else None

    def vmean(col):
        return float(lld[col].to_numpy()[voiced].mean())

    return {
        "loudness": float(lld["Loudness_sma3"].to_numpy()[voiced].mean()),
        "f0_median_st": float(np.median(vf0)),
        "f0_range_st": float(np.percentile(vf0, 90) - np.percentile(vf0, 10)),
        "f0_final_slope_st": slope,
        "jitter": vmean("jitterLocal_sma3nz"),
        "shimmer_db": vmean("shimmerLocaldB_sma3nz"),
        "hnr_db": vmean("HNRdBACF_sma3nz"),
    }


def extract(audio, sample_rate: int, text: str = "", *, num_channels: int = 1,
            response_latency_s: float | None = None, barge_in: bool = False,
            acoustic: bool = True) -> TurnFeatures:
    """Measure one caller turn. `audio` is the caller's speech for this turn only."""
    x = to_float_mono(audio, sample_rate, num_channels)
    mask = _voiced_mask(x, sample_rate)
    pauses, span = _pauses(mask)
    voiced_s = float(mask.sum() * FRAME_S)
    words = len(_WORD_RE.findall(text or ""))

    feats = TurnFeatures(
        duration_s=len(x) / sample_rate,
        voiced_s=voiced_s,
        words=words,
        words_per_sec=(words / voiced_s) if words and voiced_s >= MIN_ACOUSTIC_S else None,
        pause_count=len(pauses),
        pause_total_s=float(sum(pauses)),
        longest_pause_s=float(max(pauses, default=0.0)),
        pause_ratio=float(sum(pauses) / span) if span else 0.0,
        fillers=len(_FILLER_RE.findall(text or "")),
        response_latency_s=response_latency_s,
        barge_in=barge_in,
    )
    if acoustic and voiced_s >= MIN_ACOUSTIC_S:
        for k, v in _acoustics(x, sample_rate).items():
            setattr(feats, k, v)
    return feats
