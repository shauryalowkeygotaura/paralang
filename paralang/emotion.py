"""
emotion.py -- named emotions for ANY TTS voice, not just voice agents.

Two layers, because TTS engines split into two families:
  - Knob engines (edge-tts, Azure, Google, Polly, any SSML): emotion can only
    be *approximated* with rate / pitch / volume / pauses. EMOTIONS holds
    those numbers.
  - Instructable engines (ElevenLabs v3 audio tags, OpenAI gpt-4o-mini-tts
    `instructions`, Gemini TTS style prompts, Azure express-as styles): they
    take the emotion as words and act it out properly. EMOTIONS also holds
    the word/tag each one understands.
tts.py turns one Script into whichever form your engine wants.

Scripts can switch emotion mid-text with inline markup:
    "[warm] Namaste ji. [excited] Aaj ek bada offer hai! [calm] Koi jaldi nahi."
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .delivery import DeliveryPlanner, Prosody


@dataclass(frozen=True)
class Emotion:
    rate: int            # % at intensity 1.0
    pitch: int           # Hz
    volume: int          # %
    pause: float         # gap multiplier between sentences
    describe: str        # plain-English direction for instructable TTS
    eleven_tag: str      # ElevenLabs v3 audio tag (without brackets)
    azure_style: str     # Azure mstts:express-as style (only some voices support it)


# Tune the numbers by ear with:  python -m paralang.say --demo
EMOTIONS: dict[str, Emotion] = {
    "neutral":     Emotion(0,   0,   0, 1.0, "natural and conversational", "", ""),
    "warm":        Emotion(-3, +2,  +2, 1.1, "warm and friendly, smiling while speaking", "warmly", "friendly"),
    "excited":     Emotion(+12, +8, +8, 0.7, "excited and energetic, genuinely thrilled", "excited", "excited"),
    "cheerful":    Emotion(+6, +5,  +4, 0.85, "cheerful and upbeat", "cheerfully", "cheerful"),
    "calm":        Emotion(-10, -3, -5, 1.35, "calm, slow and reassuring", "calmly", "calm"),
    "empathetic":  Emotion(-8, -2,  -6, 1.3, "empathetic and caring, softly understanding", "sympathetic", "empathetic"),
    "serious":     Emotion(-5, -5,  0, 1.2, "serious and measured, no smile", "seriously", "serious"),
    "confident":   Emotion(-2, -3, +5, 1.0, "confident and assured, steady authority", "confidently", "assistant"),
    "curious":     Emotion(+2, +5,  0, 1.0, "curious and interested, leaning in", "curious", "chat"),
    "apologetic":  Emotion(-7, -2,  -6, 1.25, "apologetic and sincere", "apologetically", "sad"),
    "sad":         Emotion(-14, -6, -10, 1.5, "sad and subdued, heavy", "sad", "sad"),
    "whisper":     Emotion(-8,  0, -20, 1.3, "whispering, intimate and quiet", "whispers", "whispering"),
    "dramatic":    Emotion(-12, -4, +6, 1.6, "dramatic and suspenseful, slow with weight", "dramatically", "narration-professional"),
    "urgent":      Emotion(+14, +4, +8, 0.6, "urgent and fast, something matters right now", "urgently", "shouting"),
}

_TAG_RE = re.compile(r"\[([a-zA-Z_]+)(?::([0-9.]+))?\]")
_SENT_RE = re.compile(r"[^.!?…]+(?:\.\.\.|[.!?…])*", re.UNICODE)


@dataclass
class Line:
    text: str
    emotion: str
    intensity: float
    prosody: Prosody


@dataclass
class Script:
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(l.text for l in self.lines)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def parse(text: str, default: str = "neutral", intensity: float = 1.0) -> list[tuple[str, float, str]]:
    """'[warm] a. [excited:0.5] b.' -> [(warm,1.0,'a.'), (excited,0.5,'b.')].
    Unknown tags are left in the text (they may be meant for the TTS itself)."""
    out, emo, inten, pos = [], default, intensity, 0
    for m in _TAG_RE.finditer(text):
        name = m.group(1).lower()
        if name not in EMOTIONS:
            continue
        chunk = text[pos:m.start()]
        out += [(emo, inten, s) for s in split_sentences(chunk)]
        emo, inten, pos = name, float(m.group(2)) if m.group(2) else intensity, m.end()
    out += [(emo, inten, s) for s in split_sentences(text[pos:])]
    return out


def express(text: str, emotion: str = "neutral", intensity: float = 1.0, *,
            base_rate_pct: int = 0, wobble: bool = True, cues=None) -> Script:
    """Text (optionally with [emotion] markup) -> per-sentence Script.

    `cues` (from the listening side) still stack on top, so a voice agent can
    be 'warm' AND slow down for a hesitant caller at the same time."""
    if emotion not in EMOTIONS:
        raise ValueError(f"unknown emotion {emotion!r}; choose from {', '.join(EMOTIONS)}")
    planner = DeliveryPlanner(base_rate_pct=base_rate_pct, wobble=wobble)
    script = Script()
    for emo, inten, sentence in parse(text, emotion, intensity):
        p = planner.plan(sentence, cues, reply_id="script", emotion=emo, intensity=inten)
        script.lines.append(Line(sentence, emo, inten, p))
    if not script.lines:
        raise ValueError("nothing speakable in text (empty, or only [emotion] tags)")
    return script
