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


def prewarm() -> None:
    """Load openSMILE now (~12 s cold on Windows) instead of on the first turn."""
    _smile()


# ── fast path: numpy pitch tracker (default) ────────────────────────────────
# Measured 2026-10-09: openSMILE costs ~12 s cold + ~145 ms per 3 s turn, and the
# cues only use pitch + loudness. This normalised-autocorrelation tracker does
# pitch in a few ms with no startup. "full" mode keeps openSMILE for jitter /
# shimmer / HNR (stored, not used by any cue).
F0_MIN_HZ, F0_MAX_HZ = 60.0, 400.0
VOICING_THRESHOLD = 0.45     # normalised autocorrelation peak needed to call a frame voiced
_WIN_S, _HOP_S = 0.04, 0.01


PITCH_SR = 8000             # F0 <= 400 Hz needs nowhere near 16-48 kHz; ~4x less FFT work


def _decimate(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Integer-factor downsample to ~PITCH_SR with a boxcar anti-alias average.
    Speech harmonics above 4 kHz are weak, and the tracker only reads the
    periodicity, so this costs no measurable pitch accuracy (tests: <5% at 8k)."""
    factor = int(sr // PITCH_SR)
    if factor < 2:
        return x, float(sr)
    n = len(x) // factor * factor
    return x[:n].reshape(-1, factor).mean(axis=1), sr / factor


def pitch_track(x: np.ndarray, sr: int) -> np.ndarray:
    """F0 in Hz per 10 ms frame, 0 where unvoiced. Vectorised: one FFT batch."""
    x, sr = _decimate(np.asarray(x, dtype=np.float32), sr)
    win, hop = int(_WIN_S * sr), int(_HOP_S * sr)
    if len(x) < win:
        return np.zeros(0)
    n = 1 + (len(x) - win) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0])).astype(np.float32)
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = (frames ** 2).sum(axis=1)
    nfft = 1 << (2 * win - 1).bit_length()
    spec = np.fft.rfft(frames * np.hanning(win).astype(np.float32), nfft)
    ac = np.fft.irfft(np.abs(spec) ** 2, nfft)[:, :win]
    ac /= np.maximum(ac[:, :1], 1e-12)
    lo, hi = int(sr / F0_MAX_HZ), min(win - 2, int(sr / F0_MIN_HZ))
    seg = ac[:, lo:hi]
    best = seg.max(axis=1)
    # first lag reaching 90% of the best peak: avoids sub-harmonic (octave-down) picks
    lag = lo + np.argmax(seg >= 0.9 * best[:, None], axis=1)
    # walk to the local maximum, then parabolic interpolation for sub-sample precision
    for _ in range(4):
        step = ac[np.arange(n), lag + 1] > ac[np.arange(n), lag]
        lag = np.where(step & (lag < hi), lag + 1, lag)
    a, b, c = (ac[np.arange(n), lag - 1], ac[np.arange(n), lag], ac[np.arange(n), lag + 1])
    denom = a - 2 * b + c
    frac = np.where(np.abs(denom) > 1e-9, 0.5 * (a - c) / denom, 0.0)
    f0 = sr / (lag + np.clip(frac, -0.5, 0.5))
    loud = energy > (np.percentile(energy, 95) * 10 ** (-SILENCE_DB_BELOW_PEAK / 10))
    return np.where((best >= VOICING_THRESHOLD) & loud, f0, 0.0)


def _pitch_stats(f0_st: np.ndarray, hop: float) -> dict:
    voiced = f0_st > 0
    if voiced.sum() < 5:
        return {}
    t = np.arange(len(f0_st)) * hop
    vf0, vt = f0_st[voiced], t[voiced]
    tail = vt >= vt[-1] - FINAL_WINDOW_S
    slope = float(np.polyfit(vt[tail], vf0[tail], 1)[0]) if tail.sum() >= 5 else None
    return {"f0_median_st": float(np.median(vf0)),
            "f0_range_st": float(np.percentile(vf0, 90) - np.percentile(vf0, 10)),
            "f0_final_slope_st": slope}


def _loudness_db(x: np.ndarray, sr: int, mask: np.ndarray) -> float | None:
    n = max(1, int(sr * FRAME_S))
    frames = x[: len(x) // n * n].reshape(-1, n)
    if not mask.any():
        return None
    rms = np.sqrt((frames[mask[: len(frames)]] ** 2).mean(axis=1))
    return float(20 * np.log10(rms.mean() + 1e-9))


def _acoustics(x: np.ndarray, sr: int, mode: str) -> dict:
    """Pitch/voice-quality stats. Any failure degrades to {} (timing features
    still ship) because one bad turn must never break a live call."""
    try:
        if mode == "full":
            return _acoustics_opensmile(x, sr)
        hz = pitch_track(x, sr)
        st = np.where(hz > 0, 12 * np.log2(np.maximum(hz, 1e-9) / 27.5), 0.0)
        return _pitch_stats(st, _HOP_S)
    except Exception as e:  # noqa: BLE001 - boundary with a native lib
        log.warning("paralang: acoustic features skipped (%s: %s)", type(e).__name__, e)
        return {}


def voice_quality(x: np.ndarray, sr: int) -> dict:
    """openSMILE jitter / shimmer / HNR / F0 for a float mono turn. Used by the
    hybrid Analyzer in a background worker; {} on any failure."""
    try:
        return _acoustics_opensmile(x, sr)
    except Exception as e:  # noqa: BLE001 - boundary with a native lib
        log.warning("paralang: voice quality skipped (%s: %s)", type(e).__name__, e)
        return {}


def _acoustics_opensmile(x: np.ndarray, sr: int) -> dict:
    lld = _smile().process_signal(x, sr)
    f0 = lld["F0semitoneFrom27.5Hz_sma3nz"].to_numpy()   # 0 where unvoiced
    out = _pitch_stats(f0, 0.01)
    voiced = f0 > 0
    if out:
        for key, col in (("jitter", "jitterLocal_sma3nz"), ("shimmer_db", "shimmerLocaldB_sma3nz"),
                         ("hnr_db", "HNRdBACF_sma3nz")):
            out[key] = float(lld[col].to_numpy()[voiced].mean())
    return out


def extract(audio, sample_rate: int, text: str = "", *, num_channels: int = 1,
            response_latency_s: float | None = None, barge_in: bool = False,
            acoustic: bool | str = "fast") -> TurnFeatures:
    """Measure one caller turn. `audio` is the caller's speech for this turn only.
    acoustic: "fast" (numpy, default, ~ms), "full" (openSMILE, + jitter/shimmer/HNR), False."""
    mode = "fast" if acoustic is True else acoustic
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
    if mode and voiced_s >= MIN_ACOUSTIC_S:
        # loudness is always the numpy dB value so its units never depend on the mode
        feats.loudness = _loudness_db(x, sample_rate, mask)
        for k, v in _acoustics(x, sample_rate, mode).items():
            setattr(feats, k, v)
    return feats
