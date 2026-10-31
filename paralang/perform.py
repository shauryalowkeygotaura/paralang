"""
perform.py -- emotion as TEXT, for engines that only read text.

Aura has no emotion control and edge only has three knobs, but every neural
voice reacts to punctuation: "!" lifts energy and pitch, a trailing "..." lets
a line sink and fade, "?" turns the contour up. Instructable engines (Orpheus)
get the emotion as a [direction]; this is the equivalent for the rest.

Only the final punctuation of a sentence changes. Words never change, so a
voiceover still says exactly what was written and captions still match.
Fillers / interjections ("hmm", "achha") are NOT added here: that is content,
and the LLM writes it (delivery.SPEAKING_HINT).
"""
from __future__ import annotations

import re

# emotion -> (ending it gets, minimum intensity)
ENDINGS = {
    "excited":    ("!", 0.6),
    "cheerful":   ("!", 0.8),
    "urgent":     ("!", 0.6),
    "sad":        ("...", 0.6),
    "apologetic": ("...", 1.2),   # only when strongly apologetic; otherwise it sounds limp
}

_END_RE = re.compile(r"([.!?…]+)\s*$")


def shape(text: str, emotion: str, intensity: float = 1.0) -> str:
    """The sentence with its ending re-punctuated for `emotion`.
    Questions are never touched (the rising contour carries meaning)."""
    rule = ENDINGS.get(emotion)
    t = text.rstrip()
    if not rule or intensity < rule[1] or not t:
        return text
    end = _END_RE.search(t)
    current = end.group(1) if end else ""
    if "?" in current:
        return text
    if rule[0] == "!" and ("!" in current or current in ("...", "…")):
        return text            # already exclaimed, or a deliberate trail: leave the author's choice
    if rule[0] == "..." and current in ("...", "…", "!"):
        return text
    body = t[: end.start()] if end else t
    return body + rule[0]
