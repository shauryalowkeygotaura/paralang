"""
lang.py -- on-the-fly English <-> Hindi voice switching. STDLIB ONLY.

Why: Deepgram has no Hindi voice, and Andromeda reading romanized Hindi is
garbled (samples/hindi_ab, 2026-10-15: Groq Whisper-hi character error 25-42%
for Andromeda vs 0-6% for edge Swara/Neerja on the same lines; "kal khatam ho
raha hai" came back as "कार्णम हराहे"). So Hindi is spoken by ONE Hindi voice,
hi-IN Swara, which reads both Devanagari and romanized Hindi (Neerja is silent
on Devanagari). A call switches between exactly two voices, never three.

detect_lang() is voicezero's detector (Code/voicezero/tts_providers.py), kept
identical so both systems agree: any Devanagari -> Hindi; else the density of
distinctly-Hindi romanized function words, threshold 0.12, deliberately biased
toward Hindi on mixed input ("a Hindi-capable voice speaks the English words
inside a Hinglish reply just fine; the reverse is not true").

ReplyRouter adds two rules on top, for streaming agents that see one sentence
at a time:
  - sticky: once a reply turns Hindi it stays Hindi (no ping-pong mid-reply),
    released only by a clearly English sentence (RELEASE_WORDS+, no Hindi word);
  - short openers ("Achha ji.", "Okay.") under SHORT_WORDS follow the language
    the caller last spoke, since a 2-word sentence carries no signal of its own.
"""
from __future__ import annotations

import re

HINDI_VOICE = "hi-IN-SwaraNeural"
SHORT_WORDS = 4      # an English-looking sentence shorter than this mirrors the caller
RELEASE_WORDS = 6    # a Hindi reply goes back to English only on a clearly English sentence this long
THRESHOLD = 0.12

# Copied from voicezero/tts_providers.py _HINDI_ROMAN; keep the two in sync.
_HINDI_ROMAN = frozenset((
    "hai hain ho hoon hu tha thi ka ki ke ko kaa se mein mujhe muje "
    "aap aapko aapki aapse tum tumhe tumhein hum humein hamein mera meri mere "
    "apna apni nahi nahin haan haa ji kya kyun kyu kaise kaisa kab kahan kaun "
    "kar karo karna karenge karta karti raha rahi rahe gaya gayi gaye diya "
    "liya acha accha achha theek thik bilkul sahi bas abhi phir aur lekin "
    "magar matlab yaar bhai namaste namaskar dhanyavaad shukriya kripya "
    "chahiye chahta chahti sakta sakti sakte hoga hogi paisa rupaye rupay baat "
    "wala wali wale kuch sab yeh ye woh wo jo toh bhi milega chaliye"
).split())


def detect_lang(text: str) -> str:
    """'hi' or 'en' (voicezero's rule, unchanged)."""
    if any("ऀ" <= c <= "ॿ" for c in text):
        return "hi"
    toks = re.findall(r"[a-z]+", text.lower())
    if not toks:
        return "en"
    hits = sum(t in _HINDI_ROMAN for t in toks)
    return "hi" if hits / len(toks) >= THRESHOLD else "en"


class ReplyRouter:
    """Per-sentence language for one streamed reply (one instance per call)."""

    def __init__(self):
        self._reply_id: object = None
        self._reply_lang: str | None = None

    def lang(self, sentence: str, reply_id: object = None, caller_text: str = "") -> str:
        if reply_id is None or reply_id != self._reply_id:
            self._reply_id, self._reply_lang = reply_id, None
        words = re.findall(r"\w+", sentence)
        lang = detect_lang(sentence)
        if self._reply_lang == "hi":
            # sticky, released only by a clearly English sentence: RELEASE_WORDS+
            # words and not one Hindi token (long renders must not stay Hindi forever)
            clearly_en = (lang == "en" and len(words) >= RELEASE_WORDS
                          and not any(t in _HINDI_ROMAN for t in re.findall(r"[a-z]+", sentence.lower())))
            if not clearly_en:
                return "hi"
            self._reply_lang = "en"
            return "en"
        if lang == "en" and len(words) < SHORT_WORDS and caller_text:
            lang = detect_lang(caller_text)               # short opener: mirror the caller
        if lang == "hi":
            self._reply_lang = "hi"
        return lang
