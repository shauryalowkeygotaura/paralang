"""
direction.py -- understand free-form [cues] the way Fish Audio S2 does.

Fish (and Orpheus, ElevenLabs v3) take any natural-language direction in
brackets: [slightly sad], [angry][shouting], [laughing nervously], [soft tone],
[break], [emphasis] word. Those models act them. paralang's job is to make the
SAME script work on every engine, so each cue is resolved into what a
non-acting engine can do with it:

    kind      example                    knob engines (Aura/edge)       acting engines
    emotion   [slightly sad] [nervous]   EMOTIONS knobs + endings       cue passed verbatim
    tone      [whispering] [shouting]    stacked volume/rate/pitch      cue passed verbatim
    sound     [laughing] [sighing]       real clip from sounds.py       cue passed verbatim
    break     [break] [long-break]       a beat / real silence          [break] / "..."
    emphasis  [emphasis] word            *word* (beat / SSML emphasis)  as given
    unknown   [building suspense]        dropped (never read aloud)     cue passed verbatim

Resolution is a whitelist of words (no regex soup over free text): a phrase
is lower-cased, split into words, and matched against the tables below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# word -> paralang emotion (EMOTIONS keys). Exact EMOTIONS names win first.
EMOTION_WORDS = {
    "happy": "cheerful", "joyful": "cheerful", "glad": "cheerful", "upbeat": "cheerful", "bright": "cheerful",
    "delighted": "excited", "thrilled": "excited", "ecstatic": "excited", "enthusiastic": "excited",
    "energetic": "excited", "elated": "excited",
    "friendly": "warm", "kind": "warm", "welcoming": "warm", "casual": "warm",
    "sympathetic": "empathetic", "caring": "empathetic", "compassionate": "empathetic",
    "understanding": "empathetic", "sympathy": "empathetic", "comforting": "empathetic",
    "worried": "nervous", "anxious": "nervous", "uneasy": "nervous", "tense": "nervous", "unsure": "nervous",
    "hesitant": "nervous",
    "afraid": "scared", "fearful": "scared", "terrified": "scared", "panicked": "scared", "frightened": "scared",
    "shocked": "surprised", "astonished": "surprised", "amazed": "surprised", "stunned": "surprised",
    "mad": "angry", "furious": "angry", "enraged": "angry", "outraged": "angry",
    "annoyed": "frustrated", "irritated": "frustrated", "exasperated": "frustrated", "impatient": "frustrated",
    "upset": "sad", "unhappy": "sad", "depressed": "sad", "melancholic": "sad", "melancholy": "sad",
    "sorrowful": "sad", "heartbroken": "sad", "grieving": "sad", "gloomy": "sad",
    "sorry": "apologetic", "regretful": "apologetic", "remorseful": "apologetic",
    "relaxed": "calm", "peaceful": "calm", "soothing": "calm", "serene": "calm", "reassuring": "calm",
    "solemn": "serious", "grave": "serious", "stern": "serious", "formal": "serious", "formally": "serious",
    "professional": "serious", "professionally": "serious",
    "authoritative": "confident", "authoritatively": "confident", "assertive": "confident",
    "assured": "confident", "confidently": "confident",
    "intrigued": "curious", "interested": "curious", "inquisitive": "curious", "wondering": "curious",
    "optimistic": "hopeful", "resolute": "determined", "firm": "determined",
    "thankful": "grateful", "appreciative": "grateful",
    "teasing": "playful", "mischievous": "playful", "cheeky": "playful", "singsong": "playful",
    "timid": "shy", "bashful": "shy",
    "indifferent": "bored", "uninterested": "bored", "deadpan": "sarcastic", "ironic": "sarcastic",
    "dry": "sarcastic", "mocking": "sarcastic", "mock": "sarcastic",
    "awkward": "embarrassed", "wistful": "nostalgic", "loving": "tender", "affectionate": "tender",
    "gentle": "tender", "ominous": "mysterious", "menacing": "mysterious", "eerie": "mysterious",
    "suspenseful": "dramatic", "theatrical": "dramatic", "epic": "dramatic", "intense": "dramatic",
    "hurried": "urgent", "frantic": "urgent", "pleased": "proud", "satisfied": "proud",
    "warmly": "warm", "excitedly": "excited", "sadly": "sad", "angrily": "angry", "nervously": "nervous",
    "calmly": "calm", "playfully": "playful", "tenderly": "tender", "seriously": "serious",
}

# tone -> (rate %, pitch Hz, volume %) stacked on the emotion's knobs.
TONES = {
    "whispering": (-8, 0, -20), "whispers": (-8, 0, -20), "hushed": (-6, 0, -15),
    "shouting": (+6, +6, +15), "yelling": (+6, +6, +15), "screaming": (+10, +10, +15),
    "soft": (-4, 0, -10), "softly": (-4, 0, -10), "quietly": (-4, 0, -12), "breathy": (-4, 0, -8),
    "loud": (0, +2, +10), "loudly": (0, +2, +10),
    "hurry": (+15, +2, 0), "fast": (+15, 0, 0), "rapid": (+18, 0, 0), "quickly": (+12, 0, 0),
    "slow": (-12, 0, 0), "slowly": (-12, 0, 0),
    "low": (0, -5, 0), "deep": (0, -6, 0), "high": (0, +6, 0),
}

# word -> sound-bank name (sounds.py). Order-free; the LAST sound word wins
# ("laughing nervously" = laugh, "exasperated sigh" = sigh).
SOUND_WORDS = {
    "laugh": "laugh", "laughs": "laugh", "laughing": "laugh", "laughter": "laugh",
    "giggle": "laugh", "giggles": "laugh", "giggling": "laugh",
    "chuckle": "chuckle", "chuckles": "chuckle", "chuckling": "chuckle", "snicker": "chuckle",
    "sigh": "sigh", "sighs": "sigh", "sighing": "sigh",
    "gasp": "gasp", "gasps": "gasp", "gasping": "gasp",
    "breath": "breath", "breathes": "breath", "breathing": "breath", "inhale": "breath",
    "inhales": "breath", "exhale": "sigh", "exhales": "sigh",
    "panting": "breath", "pant": "breath",
    "yawn": "yawn", "yawns": "yawn", "yawning": "yawn",
    "sob": "sob", "sobs": "sob", "sobbing": "sob", "crying": "sob", "cries": "sob", "sniffle": "sob",
    "groan": "groan", "groans": "groan", "groaning": "groan",
    "ahem": "throat", "cough": "throat", "coughs": "throat", "throat": "throat",
    "hmm": "hmm", "hum": "hmm", "mm": "hmm",
}

BREAKS = {"break": 350, "pause": 350, "beat": 350, "long-break": 900, "long pause": 900,
          "long break": 900, "short pause": 250}

INTENSITY = {
    "slightly": 0.5, "somewhat": 0.6, "mildly": 0.5, "little": 0.5, "bit": 0.5, "faintly": 0.4,
    "very": 1.5, "really": 1.4, "so": 1.4, "deeply": 1.5, "highly": 1.5, "quite": 1.2,
    "extremely": 1.8, "incredibly": 1.8, "intensely": 1.8, "super": 1.7, "totally": 1.6,
}


@dataclass
class Direction:
    phrase: str                       # as written, e.g. "slightly sad"
    kind: str                         # emotion | tone | sound | break | emphasis | unknown
    emotion: str | None = None
    intensity: float | None = None
    tone: tuple[int, int, int] = (0, 0, 0)
    tones: list[str] = field(default_factory=list)
    sound: str | None = None
    pause_ms: int = 0


_WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)?")


def resolve(phrase: str, emotions: dict | None = None) -> Direction:
    """One bracket's contents -> Direction. `emotions` = EMOTIONS (passed in
    to avoid an import cycle)."""
    from .emotion import EMOTIONS
    emotions = emotions or EMOTIONS
    raw = phrase.strip()
    p = raw.lower()
    explicit = None
    m = re.fullmatch(r"([a-z_ -]+):([0-9.]+)", p)
    if m:                                   # legacy [excited:0.5]
        p, explicit = m.group(1).strip(), float(m.group(2))
    if p in emotions:
        return Direction(raw, "emotion", p, explicit if explicit is not None else None)
    if p in BREAKS:
        return Direction(raw, "break", pause_ms=BREAKS[p])
    if p == "emphasis":
        return Direction(raw, "emphasis")

    words = _WORD_RE.findall(p)
    sound = None
    for w in words:
        if w in SOUND_WORDS:
            sound = SOUND_WORDS[w]
    if sound:
        return Direction(raw, "sound", sound=sound)

    emotion = None
    for w in words:
        if w in emotions and emotion is None:
            emotion = w
        elif w in EMOTION_WORDS and emotion is None:
            emotion = EMOTION_WORDS[w]
    tone = [0, 0, 0]
    tones = []
    for w in words:
        if w in TONES:
            tones.append(w)
            for i, v in enumerate(TONES[w]):
                tone[i] += v
    intensity = explicit
    if intensity is None:
        for w in words:
            if w in INTENSITY:
                intensity = INTENSITY[w]
    if emotion:
        return Direction(raw, "emotion", emotion, intensity, tuple(tone), tones)
    if tones:
        return Direction(raw, "tone", tone=tuple(tone), tones=tones)
    return Direction(raw, "unknown")
