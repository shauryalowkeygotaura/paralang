"""
cues.py -- features + z-scores -> a few named, human-meaningful cues.

This is the ONLY place that interprets. Features are measurements; cues are
opinions about them. Every cue carries `evidence` (the numbers that fired it)
so a wrong cue can be traced to a threshold here, not to a black box.

No emotion labels ("angry", "sad"): acoustic emotion models are trained on
acted studio English and fall apart on Hinglish phone audio. These cues
describe delivery, which is what we can actually measure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .features import TurnFeatures

CUE_NAMES = ("hesitant", "long_think", "rushed", "flat", "agitated", "uncertain_rise", "barged_in")

# Absolute fallbacks: used on turns 1-2 before the caller's baseline exists,
# and as a hard floor after it. Only for things that are unambiguous in any voice.
LONG_THINK_S = 2.0          # >2 s to start answering is a think in any culture
LONG_PAUSE_S = 1.0          # a 1 s stop mid-sentence
RISE_ST_PER_S = 6.0         # final pitch rising this fast reads as a question / unsure


@dataclass
class Cue:
    name: str
    strength: float                     # 0..1
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "strength": round(self.strength, 2),
                "evidence": {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in self.evidence.items()}}


def _ramp(value: float, start: float, full: float) -> float:
    """0 at `start`, 1 at `full`, linear between, clamped. Turns a threshold into a strength."""
    if full == start:
        return 1.0 if value >= full else 0.0
    return max(0.0, min(1.0, (value - start) / (full - start)))


def derive_cues(f: TurnFeatures, z: dict[str, float]) -> list[Cue]:
    """Decide which cues this turn shows.

    `z` holds z-scores vs this caller's own baseline (keys from
    baseline.TRACKED). A key is MISSING when there is no baseline yet, which
    means "unknown", not "normal". Use z.get(k) and handle None.

    Unambiguous cues (long_think, uncertain_rise, rushed, barged_in) fire on one
    signal; the judgement cues (hesitant, flat, agitated) need two (see below).
    """
    cues: list[Cue] = []

    # long_think: slow to start answering. Absolute floor OR clearly slower than usual.
    lat = f.response_latency_s
    if lat is not None:
        s = max(_ramp(lat, LONG_THINK_S * 0.75, LONG_THINK_S * 1.5),
                _ramp(z.get("response_latency_s", 0.0), 1.5, 3.0))
        if s > 0:
            cues.append(Cue("long_think", s, {"response_latency_s": lat,
                                              "z": z.get("response_latency_s")}))

    # uncertain_rise: the turn ends going up (question / not sure / seeking approval)
    if f.f0_final_slope_st is not None:
        s = _ramp(f.f0_final_slope_st, RISE_ST_PER_S, RISE_ST_PER_S * 2.5)
        if s > 0:
            cues.append(Cue("uncertain_rise", s, {"f0_final_slope_st": f.f0_final_slope_st}))

    # rushed: talking clearly faster than they normally do (busy / wants off the call)
    zr = z.get("words_per_sec")
    if zr is not None and zr > 1.5:
        cues.append(Cue("rushed", _ramp(zr, 1.5, 3.0), {"words_per_sec": f.words_per_sec, "z": zr}))

    if f.barge_in:
        cues.append(Cue("barged_in", 1.0, {}))

    cues += _hesitant(f, z) + _flat(f, z) + _agitated(f, z)
    return cues


# ── judgement cues (decided 2026-10-19) ─────────────────────────────────────
# Design rule for all three: a cue needs AT LEAST TWO independent signals.
# One signal alone is ambiguous on Hinglish phone audio (a single pause is often
# a speaker searching for the English word; a quiet turn is often just a bad
# line), so a lone signal only fires when it is extreme, and then weakly.

SIGNAL_MIN = 0.2          # a signal counts as "present" at or above this strength
MIN_WORDS_FLAT = 3        # "haan" / "ok" are always flat; need a real sentence


def _combine(signals: dict[str, float], lone_ok: set[str] = frozenset(), lone_scale: float = 0.6):
    """Strength from several 0..1 signals: >=2 present -> their mean, boosted a
    little per extra signal; exactly 1 present -> only if it is in `lone_ok`
    and near-full strength, scaled down. Returns (strength, present_names)."""
    present = {k: v for k, v in signals.items() if v >= SIGNAL_MIN}
    if len(present) >= 2:
        mean = sum(present.values()) / len(present)
        return min(1.0, mean * (1 + 0.15 * (len(present) - 2))), sorted(present)
    if len(present) == 1:
        (k, v), = present.items()
        if k in lone_ok and v >= 0.9:
            return v * lone_scale, [k]
    return 0.0, []


def _hesitant(f: TurnFeatures, z: dict[str, float]) -> list[Cue]:
    """Unsure / reluctant: a long stop mid-turn, fillers, slower or more
    broken-up speech than this caller's normal."""
    signals = {
        "long_pause": _ramp(f.longest_pause_s, LONG_PAUSE_S * 0.6, LONG_PAUSE_S * 1.5),
        "fillers": min(1.0, 0.45 * f.fillers),
        "slower": _ramp(-(z.get("words_per_sec") or 0.0), 1.0, 2.5),
        "choppy": _ramp(z.get("pause_ratio") or 0.0, 1.5, 3.0),
    }
    # a single very long stop (>= ~1.4 s) in a real sentence still reads as hesitation
    lone = {"long_pause"} if f.words >= 3 else set()
    s, used = _combine(signals, lone_ok=lone)
    if s <= 0:
        return []
    return [Cue("hesitant", s, {"signals": used, "longest_pause_s": f.longest_pause_s,
                                "fillers": f.fillers, "z_rate": z.get("words_per_sec"),
                                "z_pause_ratio": z.get("pause_ratio")})]


def _flat(f: TurnFeatures, z: dict[str, float]) -> list[Cue]:
    """Disengaged: narrower pitch AND quieter than this caller's normal. Needs a
    baseline (z) and a real sentence; pitch range alone is common in calm
    speech, so on its own it only fires when extreme, and weakly."""
    if f.words < MIN_WORDS_FLAT or "f0_range_st" not in z:
        return []
    signals = {
        "narrow_pitch": _ramp(-z["f0_range_st"], 1.0, 2.0),
        "quieter": _ramp(-(z.get("loudness") or 0.0), 1.0, 2.0),
        "slower": _ramp(-(z.get("words_per_sec") or 0.0), 1.0, 2.0) * 0.8,
    }
    s, used = _combine(signals, lone_ok={"narrow_pitch"}, lone_scale=0.5)
    if s <= 0:
        return []
    return [Cue("flat", s, {"signals": used, "z_f0_range": z.get("f0_range_st"),
                            "z_loudness": z.get("loudness")})]


def _agitated(f: TurnFeatures, z: dict[str, float]) -> list[Cue]:
    """Louder + (higher pitch and/or faster) than this caller's normal. LOUDNESS
    IS REQUIRED: high pitch + fast without volume is usually excitement, and an
    excited caller is often a hot lead, so it must not read as agitation.
    Acoustics cannot fully separate anger from excitement; the intent scorer's
    word signals ("mehenga", "busy" vs "haan", "abhi") break the tie."""
    loud = _ramp(z.get("loudness") or 0.0, 1.5, 3.0)
    if loud < SIGNAL_MIN:
        return []
    signals = {
        "louder": loud,
        "higher_pitch": _ramp(z.get("f0_median_st") or 0.0, 1.5, 3.0),
        "faster": _ramp(z.get("words_per_sec") or 0.0, 1.5, 3.0),
    }
    s, used = _combine(signals)          # louder + at least one more
    if s <= 0:
        return []
    return [Cue("agitated", s, {"signals": used, "z_loudness": z.get("loudness"),
                                "z_f0": z.get("f0_median_st"), "z_rate": z.get("words_per_sec")})]
