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

    The obvious, unambiguous cues are implemented. The judgement calls
    (hesitant, flat, agitated) are yours: see the TODO below.
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

    # TODO(Shaurya): hesitant / flat / agitated -- the calls that shape the agent.
    #
    # Available: f.longest_pause_s, f.pause_ratio, f.fillers, f.words,
    #            z.get("pause_ratio"), z.get("words_per_sec"), z.get("loudness"),
    #            z.get("f0_range_st"), z.get("f0_median_st"), and LONG_PAUSE_S.
    # Append Cue(name, strength 0..1 via _ramp(...), evidence={...numbers used}).
    #
    # Questions to decide:
    #   hesitant -- is one long pause (f.longest_pause_s > LONG_PAUSE_S) enough
    #               on its own, or must it pair with fillers / slower speech?
    #               (A Hinglish speaker searching for an English word pauses too.)
    #   flat     -- low pitch range AND quieter than usual (disengaged), or is
    #               pitch range alone enough? Short answers ("haan") are always
    #               flat, so maybe require f.words >= 3.
    #   agitated -- louder + higher pitch + faster? How many of the three?
    #               Remember excitement (a hot lead!) looks the same acoustically.

    return cues
