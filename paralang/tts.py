"""
tts.py -- render one emotion Script for whichever TTS engine you use.

    s = express("[warm] Namaste ji. [excited] Bada offer hai!")
    to_edge(s)        -> [(text, {"rate","pitch","volume"}, pause_ms), ...]   edge-tts (free)
    to_ssml(s)        -> "<speak>...<prosody/><break/>...</speak>"           Google / Polly / any SSML
    to_ssml(s, voice="en-IN-NeerjaNeural", azure=True)                       + Azure express-as styles
    to_elevenlabs(s)  -> {"text": "[warmly] Namaste ji. [excited] ...", "voice_settings": {...}}   eleven_v3
    to_openai(s)      -> {"input": ..., "instructions": ...}                 gpt-4o-mini-tts
    to_gemini(s)      -> "Read this aloud. Namaste ji: warm...; ..."        Gemini TTS prompt
    to_orpheus(s)     -> ["[serious] ... [dramatic] And none of them... are ...", ...]  Groq Orpheus, one request per chunk
    to_aura(s)        -> [(sentence, pause_ms), ...]                         Deepgram Aura-2: gaps only

Pacing marks from the script (`|` beat, `*word*` emphasis) are rendered the way
each engine can honour them: SSML gets real <break>/<emphasis>; engines that
only read text get "..." for a beat, which every tested engine pauses on.
Measured 2026-10-08 (samples/ab): cutting Aura into fragments and slowing them
with atempo HURT intelligibility (Whisper heard "theoretical" as "Irritical"),
so the Aura target never splits below a sentence and never retimes.

Knob engines get numbers; instructable engines get words. Instructable ones
sound far more emotional: numbers can only approximate an emotion, an
instructable model actually performs it.
"""
from __future__ import annotations

import re
from xml.sax.saxutils import escape, quoteattr

from .emotion import EMOTIONS, Line, Script

BEAT_MS = 420


# A bracketed word that is not one of our EMOTIONS ([sigh], [dark chuckle],
# [whisper]) is a performance direction for the TTS itself. Orpheus and
# ElevenLabs v3 act these; every other engine would READ THEM ALOUD ("sigh"),
# so they are stripped for those. Write a direction once, every engine copes.
_DIRECTION_RE = re.compile(r"\[[a-zA-Z][a-zA-Z _-]{0,30}\]\s*")


def strip_directions(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _DIRECTION_RE.sub("", text)).strip()


def _spoken(l: Line, beat: str = "...", directions: bool = False) -> str:
    """The line as plain text with beats rendered as `beat`; emphasis dropped
    (text-only engines have no emphasis control, the beat before it carries it).
    directions=False strips [sigh]-style directions (see _DIRECTION_RE)."""
    # Chunks only ever split at a beat or an emphasis, so every join is a beat.
    out = f"{beat} ".join(chunk for chunk, _ in l.beats()).strip() or l.text
    return out if directions else strip_directions(out)


def to_edge(s: Script) -> list[tuple[str, dict, int]]:
    return [(_spoken(l), l.prosody.edge(), l.prosody.pause_before_ms) for l in s.lines]


def to_ssml(s: Script, voice: str | None = None, azure: bool = False) -> str:
    parts = []
    for l in s.lines:
        p = l.prosody
        if p.pause_before_ms:
            parts.append(f'<break time="{p.pause_before_ms}ms"/>')
        inner_text = ""
        for i, (chunk, emph) in enumerate(l.beats()):
            if i:
                inner_text += f'<break time="{BEAT_MS}ms"/>'
            chunk = strip_directions(chunk)
            inner_text += (f'<emphasis level="moderate">{escape(chunk)}</emphasis>' if emph
                           else escape(chunk))
        body = (f'<prosody rate="{p.rate_pct:+d}%" pitch="{p.pitch_hz:+d}Hz" '
                f'volume="{p.volume_pct:+d}%">{inner_text}</prosody>')
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
        said = _spoken(l, directions=True)          # eleven_v3 acts [sighs] etc. itself
        out.append(f"[{tag}] {said}" if tag and tag != last else said)
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
    return {"model": "gpt-4o-mini-tts", "input": strip_directions(s.text),
            "instructions": _direction(s) + " Natural human pauses between sentences."}


def to_gemini(s: Script) -> str:
    return f"{_direction(s)} Read the following aloud:\n{strip_directions(s.text)}"


def to_orpheus(s: Script, max_chars: int = 600) -> list[str]:
    """Groq `canopylabs/orpheus-v1-english` input, one string per request.

    Orpheus ACTS bracketed directions itself, so the emotion goes in as words
    and no knob is touched. Sentences are packed into chunks of up to
    `max_chars`: the free tier allows 10 requests/minute, so one request per
    sentence would take an hour on a 17-minute script. Each chunk restates its
    opening direction, because a chunk is a separate request with no memory.
    """
    chunks: list[str] = []
    cur, cur_emo = "", None
    for l in s.lines:
        said = _spoken(l, directions=True)          # Orpheus acts [sigh], [dark chuckle]...
        tag = "" if l.emotion == "neutral" else f"[{l.emotion}] "
        if cur and len(cur) + len(said) + len(tag) + 1 > max_chars:
            chunks.append(cur.strip())
            cur, cur_emo = "", None
        if not cur:
            cur, cur_emo = tag + said, l.emotion
        elif l.emotion != cur_emo:
            cur += " " + tag + said
            cur_emo = l.emotion
        else:
            cur += " " + said
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def to_aura(s: Script) -> list[tuple[str, int]]:
    """Deepgram Aura-2 has no prosody controls: pacing = the gaps between
    sentences (from the planner) plus "..." beats it pauses on. One request per
    sentence, audio never retimed (retiming fragments hurt clarity)."""
    return [(_spoken(l), l.prosody.pause_before_ms) for l in s.lines]
