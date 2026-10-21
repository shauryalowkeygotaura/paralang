"""
backchannel.py -- "haan", "hmm", "achha" while the agent talks is LISTENING, not
an interruption (v0.3 #1). Framework-agnostic rule; the Pipecat strategy lives in
adapters/pipecat.py (BackchannelAwareStartStrategy).

Before this, any sound from the caller while Riya spoke cut her off, so a caller
politely saying "haan ji" mid-sentence made her stop and wait: the most robotic
thing a phone agent does.

Rule: a caller sound during agent speech is a backchannel when it is SHORT
(VAD stopped within HOLD_S) AND its transcript is only backchannel tokens
(at most MAX_TOKENS of them). Anything longer, or any other word, interrupts.
"""
from __future__ import annotations

import re

HOLD_S = 0.5        # speech still going this long into the agent's turn = real interruption
MAX_TOKENS = 3

# English, romanized Hindi, and the Devanagari forms Whisper writes with a hi hint.
BACKCHANNEL = frozenset("""
haan han haa ha hmm hm hmmm mm mhm uh-huh uhhuh ok okay okk achha acha accha theek thik hai
ji jee right yes yeah yep sure correct bilkul sahi
हां हाँ हम्म हम अच्छा जी ठीक है हाँजी बिल्कुल सही ओके
""".split())


_PUNCT = re.compile(r"[\s.,!?;:\"'()\[\]…।॥]+")


def is_backchannel(text: str) -> bool:
    # split on whitespace/punctuation, NOT \w: Python's \w drops Devanagari
    # combining marks (matras, anusvara), so "हाँ" would never match
    toks = [t for t in _PUNCT.split(text.lower()) if t]
    return 0 < len(toks) <= MAX_TOKENS and all(t in BACKCHANNEL for t in toks)
