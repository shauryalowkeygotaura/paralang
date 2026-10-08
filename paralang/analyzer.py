"""
analyzer.py -- the one object every adapter talks to.

    an = Analyzer()
    res = an.analyze_turn(pcm, 16000, "haan... theek hai", response_latency_s=2.4)
    res.tag       -> "[tone: long pause before answering]"
    res.to_dict() -> the JSON contract (schema.json)

One Analyzer per call: it owns that caller's baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .baseline import Baseline
from .cues import Cue, derive_cues
from .features import TurnFeatures, extract

SCHEMA_VERSION = "1"

# Spoken-English phrasing of each cue for the LLM. Short: it is read every turn.
PHRASE = {
    "long_think":     "long pause before answering",
    "hesitant":       "hesitant",
    "rushed":         "speaking fast",
    "flat":           "flat, low energy",
    "agitated":       "agitated",
    "uncertain_rise": "sounds unsure",
    "barged_in":      "interrupted you",
}
TAG_MIN_STRENGTH = 0.4   # weaker cues go in the JSON but not in the LLM's prompt


@dataclass
class TurnResult:
    turn: int
    text: str
    features: TurnFeatures
    z: dict[str, float]
    cues: list[Cue]
    baseline_ready: bool
    elapsed_ms: float
    ts: float = field(default_factory=time.time)

    @property
    def tag(self) -> str:
        strong = sorted((c for c in self.cues if c.strength >= TAG_MIN_STRENGTH),
                        key=lambda c: -c.strength)
        if not strong:
            return ""
        return "[tone: " + ", ".join(PHRASE.get(c.name, c.name) for c in strong) + "]"

    def annotate(self, text: str | None = None) -> str:
        """Transcript text with the tag prefixed (what the LLM should see)."""
        text = self.text if text is None else text
        return f"{self.tag} {text}".strip() if self.tag else text

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "turn": self.turn,
            "ts": round(self.ts, 3),
            "text": self.text,
            "tag": self.tag,
            "cues": [c.to_dict() for c in self.cues],
            "features": self.features.to_dict(),
            "z": {k: round(v, 3) for k, v in self.z.items()},
            "baseline_ready": self.baseline_ready,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class Analyzer:
    def __init__(self, warmup_turns: int = 3, acoustic: bool = True):
        self.baseline = Baseline(warmup=warmup_turns)
        self.acoustic = acoustic
        self.turns: list[TurnResult] = []

    def features(self, audio, sample_rate: int, text: str = "", **kw) -> TurnFeatures:
        """Measurement only (no state, safe in a worker thread). Lets live
        adapters run the slow part before the transcript even arrives."""
        return extract(audio, sample_rate, text, acoustic=self.acoustic, **kw)

    def score(self, feats: TurnFeatures, text: str = "", elapsed_ms: float = 0.0) -> TurnResult:
        """Stateful part: z against baseline, derive cues, then grow the baseline."""
        z = self.baseline.z(feats)
        cues = derive_cues(feats, z)
        res = TurnResult(turn=len(self.turns) + 1, text=text, features=feats, z=z, cues=cues,
                         baseline_ready=self.baseline.frozen, elapsed_ms=elapsed_ms)
        self.baseline.update(feats)
        self.turns.append(res)
        return res

    def analyze_turn(self, audio, sample_rate: int, text: str = "", **kw) -> TurnResult:
        t0 = time.perf_counter()
        feats = self.features(audio, sample_rate, text, **kw)
        return self.score(feats, text, (time.perf_counter() - t0) * 1000)

    def call_signals(self, min_strength: float = 0.5, min_turns: int = 1) -> list[str]:
        """Distinct cue names seen across the call as scorer-friendly phrases
        ("tone: hesitant"). A cue counts once per call, however many turns."""
        counts: dict[str, int] = {}
        for t in self.turns:
            for c in t.cues:
                if c.strength >= min_strength:
                    counts[c.name] = counts.get(c.name, 0) + 1
        return [f"tone: {n}" for n, k in counts.items() if k >= min_turns]
