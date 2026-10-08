"""
tts.py -- render one emotion Script for whichever TTS engine you use.

    s = express("[warm] Namaste ji. [excited] Bada offer hai!")
    to_edge(s)        -> [(text, {"rate","pitch","volume"}, pause_ms), ...]   edge-tts (free)
    to_ssml(s)        -> "<speak>...<prosody/><break/>...</speak>"           Google / Polly / any SSML
    to_ssml(s, voice="en-IN-NeerjaNeural", azure=True)                       + Azure express-as styles
    to_elevenlabs(s)  -> {"text": "[warmly] Namaste ji. [excited] ...", "voice_settings": {...}}   eleven_v3
    to_openai(s)      -> {"input": ..., "instructions": ...}                 gpt-4o-mini-tts
    to_gemini(s)      -> "Read this aloud. Namaste ji: warm...; ..."        Gemini TTS prompt

Knob engines get numbers; instructable engines get words. Instructable ones
sound far more emotional: numbers can only approximate an emotion, an
instructable model actually performs it.
"""
from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from .emotion import EMOTIONS, Script


def to_edge(s: Script) -> list[tuple[str, dict, int]]:
    return [(l.text, l.prosody.edge(), l.prosody.pause_before_ms) for l in s.lines]


def to_ssml(s: Script, voice: str | None = None, azure: bool = False) -> str:
    parts = []
    for l in s.lines:
        p = l.prosody
        if p.pause_before_ms:
            parts.append(f'<break time="{p.pause_before_ms}ms"/>')
        body = (f'<prosody rate="{p.rate_pct:+d}%" pitch="{p.pitch_hz:+d}Hz" '
                f'volume="{p.volume_pct:+d}%">{escape(l.text)}</prosody>')
        style = EMOTIONS[l.emotion].azure_style
        if azure and style:
            degree = f"{max(0.01, min(2.0, l.intensity)):.2f}"
            body = (f'<mstts:express-as style={quoteattr(style)} styledegree="{degree}">'
                    f"{body}</mstts:express-as>")
        parts.append(body)
    inner = "".join(parts)
    if voice:
        inner = f"<voice name={quoteattr(voice)}>{inner}</voice>"
    ns = (' xmlns:mstts="https://www.w3.org/2001/mstts"' if azure else "")
    return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"{ns} '
            f'xml:lang="en-IN">{inner}</speak>')


def to_elevenlabs(s: Script) -> dict:
    """eleven_v3 audio tags inline; voice_settings tuned to the dominant emotion.
    Lower stability = more expressive (ElevenLabs' own guidance)."""
    out, last = [], None
    for l in s.lines:
        tag = EMOTIONS[l.emotion].eleven_tag
        out.append(f"[{tag}] {l.text}" if tag and tag != last else l.text)
        last = tag or last
    dominant = max((l.emotion for l in s.lines), key=lambda e: sum(x.emotion == e for x in s.lines),
                   default="neutral")
    expressive = dominant not in ("neutral", "calm", "serious", "confident")
    return {"text": " ".join(out), "model_id": "eleven_v3",
            "voice_settings": {"stability": 0.3 if expressive else 0.6, "similarity_boost": 0.75,
                               "style": 0.6 if expressive else 0.2}}


def _direction(s: Script) -> str:
    segs, cur = [], None
    for l in s.lines:
        if l.emotion != cur:
            segs.append((l.emotion, l.intensity, l.text))
            cur = l.emotion
    if len(segs) == 1:
        e, k, _ = segs[0]
        return f"Speak {_strength(k)}{EMOTIONS[e].describe}."
    steps = "; ".join(f'from "{t[:40]}" be {_strength(k)}{EMOTIONS[e].describe}' for e, k, t in segs)
    return f"Shift tone through the text: {steps}."


def _strength(k: float) -> str:
    return "slightly " if k < 0.6 else ("very " if k > 1.3 else "")


def to_openai(s: Script) -> dict:
    return {"model": "gpt-4o-mini-tts", "input": s.text,
            "instructions": _direction(s) + " Natural human pauses between sentences."}


def to_gemini(s: Script) -> str:
    return f"{_direction(s)} Read the following aloud:\n{s.text}"
