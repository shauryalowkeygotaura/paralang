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


def parts(text: str) -> list[tuple[str, object]]:
    """A spoken line (directions kept) -> what a knob engine does, in order:
    ("text", str) to synthesize, ("sound", name) to splice from sounds.py,
    ("pause", ms) of real silence. Emotion/tone cues were already applied as
    prosody, unknown cues are dropped: nothing in brackets is ever read aloud."""
    from .direction import resolve
    from .emotion import _TAG_RE
    out: list[tuple[str, object]] = []
    pos = 0

    def add_text(t: str) -> None:
        t = re.sub(r"\s{2,}", " ", t).strip()
        if any(c.isalnum() for c in t):
            out.append(("text", t))
        elif t:                                     # stray punctuation joins the last words
            for j in range(len(out) - 1, -1, -1):
                if out[j][0] == "text":
                    out[j] = ("text", out[j][1] + t)
                    break

    for m in _TAG_RE.finditer(text):
        add_text(text[pos:m.start()])
        d = resolve(m.group(0)[1:-1])
        if d.kind == "sound":
            out.append(("sound", d.sound))
        elif d.kind == "break" and d.pause_ms:
            out.append(("pause", d.pause_ms))
        pos = m.end()
    add_text(text[pos:])
    return out


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
            chunk = escape(strip_directions(chunk.replace("[long-break]", "\x00")))
            chunk = chunk.replace("\x00", '<break time="900ms"/>')
            inner_text += (f'<emphasis level="moderate">{chunk}</emphasis>' if emph else chunk)
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
        tag = l.cue or EMOTIONS[l.emotion].eleven_tag
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
        # the cue as the author wrote it ("slightly sad", "angry][shouting"), else our label
        tag = f"[{l.cue}] " if l.cue else ("" if l.emotion == "neutral" else f"[{l.emotion}] ")
        said = said.replace("[long-break]", "...")
        if cur and len(cur) + len(said) + len(tag) + 1 > max_chars:
            chunks.append(cur.strip())
            cur, cur_emo = "", None
        if not cur:
            cur, cur_emo = tag + said, (l.emotion, l.cue)
        elif (l.emotion, l.cue) != cur_emo:
            cur += " " + tag + said
            cur_emo = (l.emotion, l.cue)
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


# paralang emotion -> Fish Audio S2 cue. Fish takes free-form [natural
# language]; these are its documented tags where one exists (happy, sad,
# excited, calm, confident, whispering, "in a hurry tone", "soft tone").
FISH_CUES = {
    "neutral":    "",
    "warm":       "warm and friendly",
    "excited":    "excited",
    "cheerful":   "happy",
    "calm":       "calm",
    "empathetic": "empathetic][soft tone",
    "serious":    "serious",
    "confident":  "confident",
    "curious":    "curious",
    "apologetic": "apologetic][soft tone",
    "sad":        "sad",
    "whisper":    "whispering",
    "dramatic":   "dramatic",
    "urgent":     "in a hurry tone",
}
_FISH_EMPH_RE = re.compile(r"\*([^*\n]+)\*")


def _fish_cue(emotion: str, intensity: float) -> str:
    cue = FISH_CUES.get(emotion, emotion)
    if not cue:
        return ""
    # Fish reads intensity words: "[slightly sad]", "[extremely angry]"
    first, sep, rest = cue.partition("][")
    if intensity < 0.6:
        first = f"slightly {first}"
    elif intensity > 1.3:
        first = f"very {first}"
    return f"[{first}{sep}{rest}] " if sep else f"[{first}] "


def _fish_line(l: Line) -> str:
    """One sentence for Fish: `|` -> [break], *word* -> [emphasis] word,
    [laughing]-style directions kept (Fish acts them)."""
    t = l.marked or l.text
    t = _FISH_EMPH_RE.sub(r"[emphasis] \1", t)
    t = re.sub(r"\s*\|\s*", " [break] ", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def fish_chunks(s: Script, max_chars: int = 1200) -> list[tuple[str, list[Line]]]:
    """[(request text, the lines in it)]. The cue goes at the start of the
    sentence it controls (Fish: distant cues lose effect), on every emotion
    change, and is restated at the start of each request (no memory across
    requests). Rate/pitch knobs are NOT used: Fish performs the emotion."""
    chunks: list[tuple[str, list[Line]]] = []
    cur, lines, cur_emo = "", [], None
    for l in s.lines:
        said = _fish_line(l)
        key = (l.emotion, round(l.intensity, 2), l.cue)
        cue = f"[{l.cue}] " if l.cue else _fish_cue(l.emotion, l.intensity)
        if cur and len(cur) + len(said) + len(cue) + 1 > max_chars:
            chunks.append((cur.strip(), lines))
            cur, lines, cur_emo = "", [], None
        if not cur:
            cur, cur_emo = cue + said, key
        elif key != cur_emo:
            cur += " " + cue + said
            cur_emo = key
        else:
            cur += " " + said
        lines.append(l)
    if cur.strip():
        chunks.append((cur.strip(), lines))
    return chunks


def to_fish(s: Script, max_chars: int = 1200) -> list[str]:
    """Fish Audio S2.1 input, one string per request (see fish_chunks)."""
    return [t for t, _ in fish_chunks(s, max_chars)]
