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
    # 2026-11-05: Fish Audio S2's emotion set, so its scripts work on every engine
    "angry":       Emotion(+6,  -2, +10, 0.8, "angry, sharp and forceful", "angry", "angry"),
    "frustrated":  Emotion(+4,  -2, +6, 0.85, "frustrated and exasperated", "frustrated", "angry"),
    "nervous":     Emotion(+8,  +4, -6, 0.8, "nervous and unsure, a little shaky", "nervously", "fearful"),
    "scared":      Emotion(+10, +6, -8, 0.8, "scared and breathless", "scared", "terrified"),
    "surprised":   Emotion(+6, +10, +6, 0.9, "surprised, caught off guard", "surprised", "cheerful"),
    "sarcastic":   Emotion(-6,  -2,  0, 1.1, "sarcastic, dry and knowing", "sarcastically", "unfriendly"),
    "hopeful":     Emotion(-2,  +4,  0, 1.05, "hopeful and quietly optimistic", "hopeful", "hopeful"),
    "determined":  Emotion(-2,  -3, +8, 0.95, "determined, firm and resolute", "determined", "serious"),
    "disappointed": Emotion(-8, -4, -6, 1.3, "disappointed and let down", "disappointed", "sad"),
    "proud":       Emotion(-2,  +2, +6, 1.05, "proud and glowing", "proudly", "cheerful"),
    "relieved":    Emotion(-6,  +2, -2, 1.2, "relieved, letting the tension go", "relieved", "friendly"),
    "grateful":    Emotion(-4,  +3,  0, 1.15, "grateful and sincere", "gratefully", "friendly"),
    "amused":      Emotion(+2,  +4, +2, 1.0, "amused, holding back a laugh", "amused", "cheerful"),
    "playful":     Emotion(+6,  +6, +4, 0.9, "playful and teasing", "playfully", "cheerful"),
    "shy":         Emotion(-6,  +2, -12, 1.2, "shy and soft-spoken", "shyly", "embarrassed"),
    "embarrassed": Emotion(-4,  +3, -10, 1.1, "embarrassed and a little awkward", "embarrassed", "embarrassed"),
    "bored":       Emotion(-8,  -5, -6, 1.2, "bored and flat, unimpressed", "bored", ""),
    "nostalgic":   Emotion(-10, -2, -4, 1.4, "nostalgic and wistful", "wistfully", "sad"),
    "tender":      Emotion(-10,  0, -10, 1.35, "tender and gentle, close and loving", "tenderly", "gentle"),
    "mysterious":  Emotion(-12, -4, -8, 1.5, "mysterious, low and slow, drawing you in", "mysteriously", "whispering"),
}

# Any bracketed cue: [warm], [excited:0.5], [slightly sad], [laughing nervously].
# group(1) = the phrase, group(2) = an explicit :intensity.
_TAG_RE = re.compile(r"\[([a-zA-Z][a-zA-Z _-]{0,40}?)(?::([0-9.]+))?\]")
# "।" / "॥" (danda) end Hindi sentences; without them a Hindi sentence and the
# English one after it became one line and the Hindi voice read both.
_SENT_RE = re.compile(r"[^.!?…।॥]+(?:\.\.\.|[.!?…।॥])*", re.UNICODE)


_EMPH_RE = re.compile(r"\*([^*\n]+)\*")


def clean(text: str) -> str:
    """Strip pacing marks: `|` (a beat) and `*word*` (emphasis)."""
    text = _EMPH_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text.replace("|", " ")).strip()


@dataclass
class Line:
    text: str            # clean text, what every engine can read
    emotion: str
    intensity: float
    prosody: Prosody
    # Same sentence WITH pacing marks: `|` = a beat inside the sentence, `*word*`
    # = emphasis. Sentence gaps alone cannot make one word land; these can.
    # Sound / long-break / unknown [cues] stay in it for the engines to realize.
    marked: str = ""
    # The cue words as written ("slightly sad", "angry][shouting"): acting
    # engines (Orpheus, Fish, ElevenLabs) get these verbatim, not our label.
    cue: str = ""

    def beats(self) -> list[tuple[str, bool]]:
        """[(chunk, emphasised)] in order, split at beats. Emphasis also opens a beat."""
        out: list[tuple[str, bool]] = []
        for piece in (self.marked or self.text).split("|"):
            for i, part in enumerate(_EMPH_RE.split(piece)):
                part = part.strip()
                if not part:
                    continue
                # Bare punctuation belongs to the chunk before it, never its own
                # beat: "*theoretical*." must not become "theoretical... ."
                if out and not any(c.isalnum() for c in part):
                    prev, emph = out[-1]
                    out[-1] = (prev + part, emph)
                    continue
                out.append((part, i % 2 == 1))
        return out


@dataclass
class Script:
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(l.text for l in self.lines)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


_EMPHASIS_CUE_RE = re.compile(r"\[emphasis\]\s*([\w'-]+)", re.IGNORECASE)
_BREAK_CUE_RE = re.compile(r"\[(?:break|pause|beat)\]", re.IGNORECASE)


@dataclass
class Segment:
    emotion: str
    intensity: float
    sentence: str
    cue: str = ""
    tone: tuple = (0, 0, 0)


def segments(text: str, default: str = "neutral", intensity: float = 1.0) -> list[Segment]:
    """Text with any [cues] -> per-sentence Segments.

    Emotion cues ([warm], [slightly sad], [nervous]) and tone cues
    ([whispering], [soft tone]) switch the delivery for the text after them;
    adjacent cues stack ("[angry][shouting]"). [break] becomes a `|` beat and
    "[emphasis] word" becomes *word*. Sound, long-break and unknown cues stay
    in the sentence for the engine to realize (see direction.py)."""
    from .direction import resolve
    text = _EMPHASIS_CUE_RE.sub(r"*\1*", text)
    text = _BREAK_CUE_RE.sub(" | ", text)
    out: list[Segment] = []
    emo, inten, cue, tone, pos, last_end = default, intensity, "", (0, 0, 0), 0, -1
    for m in _TAG_RE.finditer(text):
        d = resolve(m.group(0)[1:-1])
        if d.kind not in ("emotion", "tone"):
            continue
        out += [Segment(emo, inten, x, cue, tone) for x in split_sentences(text[pos:m.start()])]
        stacked = last_end >= 0 and not text[last_end:m.start()].strip()
        phrase = d.phrase.split(":")[0].strip()          # "[excited:0.5]" -> "excited"
        if stacked and cue:
            cue, tone = f"{cue}][{phrase}", tuple(a + b for a, b in zip(tone, d.tone))
        elif stacked and d.kind == "tone" and emo != default:
            cue, tone = f"{emo}][{phrase}", d.tone       # "[sad][whispering]": keep both words
        else:
            # a bare paralang emotion name needs no verbatim cue: engines render
            # their own wording (with intensity) from Line.emotion
            cue = "" if phrase.lower() in EMOTIONS else phrase
            tone = d.tone
        if d.kind == "emotion":
            emo, inten = d.emotion, d.intensity if d.intensity is not None else intensity
        pos = last_end = m.end()
    out += [Segment(emo, inten, x, cue, tone) for x in split_sentences(text[pos:])]
    return out


def parse(text: str, default: str = "neutral", intensity: float = 1.0) -> list[tuple[str, float, str]]:
    """'[warm] a. [excited:0.5] b.' -> [(warm,1.0,'a.'), (excited,0.5,'b.')].
    Free-form cues resolve too ([slightly sad] -> sad, 0.5). Unknown tags are
    left in the text (they may be meant for the TTS itself)."""
    return [(g.emotion, g.intensity, g.sentence) for g in segments(text, default, intensity)]


def express(text: str, emotion: str = "neutral", intensity: float = 1.0, *,
            base_rate_pct: int = 0, wobble: bool = True, cues=None) -> Script:
    """Text (optionally with [emotion] markup) -> per-sentence Script.

    `cues` (from the listening side) still stack on top, so a voice agent can
    be 'warm' AND slow down for a hesitant caller at the same time."""
    if emotion not in EMOTIONS:
        raise ValueError(f"unknown emotion {emotion!r}; choose from {', '.join(EMOTIONS)}")
    planner = DeliveryPlanner(base_rate_pct=base_rate_pct, wobble=wobble)
    script = Script()
    from .delivery import _clamp
    for g in segments(text, emotion, intensity):
        plain = clean(g.sentence)
        if not plain:
            continue
        if not _TAG_RE.sub("", plain).strip(" .,!?…|"):
            # only cues, no words (a trailing "[laughing]"): ride on the previous line
            if script.lines:
                prev = script.lines[-1]
                prev.text, prev.marked = f"{prev.text} {plain}", f"{prev.marked} {g.sentence.strip()}"
            continue
        p = planner.plan(plain, cues, reply_id="script", emotion=g.emotion, intensity=g.intensity)
        if any(g.tone):
            p = type(p)(_clamp(p.rate_pct + g.tone[0], "rate"), _clamp(p.pitch_hz + g.tone[1], "pitch"),
                        _clamp(p.volume_pct + g.tone[2], "volume"), p.pause_before_ms)
        script.lines.append(Line(plain, g.emotion, g.intensity, p, marked=g.sentence, cue=g.cue))
    if not script.lines:
        raise ValueError("nothing speakable in text (empty, or only [emotion] tags)")
    return script
