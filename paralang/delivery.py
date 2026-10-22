"""
delivery.py -- the agent's OWN paralanguage: how to say each sentence.

The listening side (features/cues) hears how the caller sounds. This side
decides how the agent should sound back, per sentence:

  1. React to the caller. People adjust without thinking: you slow down and
     soften for someone unsure, speed up for someone in a hurry, calm down
     for someone agitated, brighten for someone flat. STYLE is that table.
  2. Never sound like a loop. A TTS voice saying every sentence at the exact
     same tempo and pitch is the #1 "robot" tell. Each sentence gets a small,
     deterministic wobble (same text -> same wobble, so tests are stable).
  3. Breathe between sentences. Humans leave a gap between thoughts; TTS
     glues sentences together. pause_before_ms inserts real silence.
     The FIRST sentence of a reply gets none: LLM latency already is the gap.

Output is TTS-agnostic numbers; .edge() renders the edge-tts/Azure form.
Content-level humanness (fillers like "achha", short sentences) is the LLM's
job, steered by SPEAKING_HINT; this module only shapes delivery.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .cues import Cue

# cue -> (rate %, pitch Hz, volume %, pause multiplier) at full strength.
# Tune here. Small numbers: +-10% rate is already clearly audible.
STYLE = {
    "hesitant":       (-8, -2, -5, 1.4),   # patient, gentle, room to think
    "uncertain_rise": (-6,  0, -3, 1.3),
    "long_think":     (-6, -2, -3, 1.3),
    "rushed":         (+8,  0,  0, 0.6),   # match their pace, get to the point
    "barged_in":      (+4,  0,  0, 0.7),
    "flat":           (+3, +6, +5, 0.9),   # a little brighter to re-engage
    "agitated":       (-10, -6, -10, 1.3), # calm, lower, softer: de-escalate
    "noisy":          (-5,  0, +10, 1.2),  # slower and louder, so it carries over the noise
}

# For instructable engines (Orpheus, ElevenLabs...) a cue becomes a named
# emotion they ACT, instead of rate/pitch numbers. Same intent as STYLE.
CUE_EMOTION = {
    "hesitant":       "empathetic",
    "uncertain_rise": "warm",
    "long_think":     "calm",
    "rushed":         "confident",   # brisk and to the point
    "barged_in":      "calm",
    "flat":           "cheerful",
    "agitated":       "calm",
}
CUE_EMOTION_MIN = 0.4


def emotion_for(cues: list[Cue] | None, base: str = "warm") -> str:
    """The emotion an instructable TTS should act for the caller's last turn:
    the strongest cue at or above CUE_EMOTION_MIN, else `base`. Agitated
    always wins: an angry caller who also talks fast must get calm, not the
    brisk 'confident' that rushed maps to (a tie used to go to rushed)."""
    if any(c.name == "agitated" and c.strength >= CUE_EMOTION_MIN for c in cues or []):
        return CUE_EMOTION["agitated"]
    best = max((c for c in cues or [] if c.name in CUE_EMOTION and c.strength >= CUE_EMOTION_MIN),
               key=lambda c: c.strength, default=None)
    return CUE_EMOTION[best.name] if best else base


BASE_PAUSE_MS = {"?": 380, ".": 320, "!": 300, "…": 450, ",": 200}
DEFAULT_PAUSE_MS = 260
QUESTION_PITCH_HZ = 4       # questions sit slightly higher
WOBBLE_RATE_PCT = 3         # +- per-sentence tempo variation
WOBBLE_PITCH_HZ = 2
LIMITS = {"rate": (-20, 25), "pitch": (-12, 12), "volume": (-20, 15)}

SPEAKING_HINT = (
    "Talk like a real person on a phone call, not a script: short sentences, one idea each; "
    "sometimes open with a natural acknowledgement (achha, haan ji, ji bilkul, hmm, samjha) but "
    "never the same one twice in a row; use '...' where a person would pause to think. "
    "No lists, no symbols, no reading out formatting."
)


@dataclass
class Prosody:
    rate_pct: int = 0
    pitch_hz: int = 0
    volume_pct: int = 0
    pause_before_ms: int = 0

    def edge(self) -> dict:
        """kwargs for edge_tts.Communicate (also valid Azure SSML prosody values)."""
        return {"rate": f"{self.rate_pct:+d}%", "pitch": f"{self.pitch_hz:+d}Hz",
                "volume": f"{self.volume_pct:+d}%"}


def _wobble(text: str) -> tuple[float, float]:
    """Stable pseudo-random (-1..1, -1..1) from the sentence text."""
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return (h[0] / 127.5 - 1, h[1] / 127.5 - 1)


def _clamp(v: float, key: str) -> int:
    lo, hi = LIMITS[key]
    return int(round(max(lo, min(hi, v))))


class DeliveryPlanner:
    """One per call. Feed it the caller's latest cues; ask it per sentence."""

    def __init__(self, base_rate_pct: int = 0, base_pitch_hz: int = 0, wobble: bool = True):
        self.base_rate_pct = base_rate_pct
        self.base_pitch_hz = base_pitch_hz
        self.wobble = wobble
        self._reply_id: object = None
        self._prev_end = "."

    def plan(self, sentence: str, cues: list[Cue] | None = None, reply_id: object = None,
             emotion: str | None = None, intensity: float = 1.0) -> Prosody:
        """Prosody for one sentence. `reply_id` identifies the reply/script this
        sentence belongs to (Pipecat's context_id); a new id = first sentence.
        `emotion` (see emotion.EMOTIONS) sets the base colour; cues stack on top."""
        first = reply_id is None or reply_id != self._reply_id
        self._reply_id = reply_id

        rate, pitch, vol, pause_mul = float(self.base_rate_pct), float(self.base_pitch_hz), 0.0, 1.0
        if emotion:
            from .emotion import EMOTIONS  # lazy: emotion imports this module
            e = EMOTIONS.get(emotion)
            if e is None:
                raise ValueError(f"unknown emotion {emotion!r}")
            k = max(0.0, min(2.0, intensity))
            rate, pitch, vol = rate + e.rate * k, pitch + e.pitch * k, vol + e.volume * k
            pause_mul *= 1 + (e.pause - 1) * k
        for c in cues or []:
            st = STYLE.get(c.name)
            if not st:
                continue
            s = max(0.0, min(1.0, c.strength))
            rate += st[0] * s
            pitch += st[1] * s
            vol += st[2] * s
            pause_mul *= 1 + (st[3] - 1) * s

        text = sentence.strip()
        if text.endswith("?"):
            pitch += QUESTION_PITCH_HZ
        if self.wobble and text:
            wr, wp = _wobble(text)
            rate += wr * WOBBLE_RATE_PCT
            pitch += wp * WOBBLE_PITCH_HZ

        pause = 0
        if not first:
            # Gap sized by how the PREVIOUS sentence ended: a question earns a
            # longer beat than a statement, a trailing "..." the longest.
            pause = int(BASE_PAUSE_MS.get(self._prev_end, DEFAULT_PAUSE_MS) * pause_mul)
        self._prev_end = "…" if text.endswith("...") else text[-1:]
        return Prosody(_clamp(rate, "rate"), _clamp(pitch, "pitch"), _clamp(vol, "volume"), pause)
