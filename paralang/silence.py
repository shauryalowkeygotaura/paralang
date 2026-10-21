"""
silence.py -- what to do when the caller goes quiet (v0.3 #4). Framework-agnostic.

A fixed "are you there?" timer is wrong for half the callers: a slow, thoughtful
speaker gets nagged, a fast one waits in dead air. So the wait scales with THIS
caller's own typical response latency (from the Analyzer's turns):

    timeout = clamp(FACTOR x median response latency, MIN_S, MAX_S)   (DEFAULT_S until 2 turns)

First silence -> a gentle nudge in the caller's language. Second consecutive
silence -> a polite close, then the agent ends the call. Any caller speech resets.

Pipecat 1.12 already has the timer (LLMUserAggregatorParams.user_idle_timeout,
event "on_user_turn_idle", UserIdleTimeoutUpdateFrame); this module only decides
the timeout and the words. See jio rt_agent.build_task for the wiring.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .lang import detect_lang

FACTOR = 2.5
MIN_S, MAX_S, DEFAULT_S = 4.0, 9.0, 6.0

NUDGE = {
    "hi": "Hello ji, aap sun pa rahe hain?",
    "en": "Hello, are you still there?",
}
CLOSE = {
    "hi": "Lagta hai line kat gayi. Main baad mein dobara call karti hoon. Dhanyavaad!",
    "en": "Looks like we got cut off. I'll try you again later. Thank you!",
}


@dataclass
class SilenceAction:
    say: str
    end_call: bool


class SilencePolicy:
    def __init__(self, analyzer=None, default_lang: str = "hi"):
        self.analyzer = analyzer
        self.default_lang = default_lang
        self.strikes = 0

    def timeout_s(self) -> float:
        lats = [t.features.response_latency_s for t in getattr(self.analyzer, "turns", [])
                if t.features.response_latency_s is not None]
        if len(lats) < 2:
            return DEFAULT_S
        return max(MIN_S, min(MAX_S, FACTOR * statistics.median(lats)))

    def caller_lang(self) -> str:
        turns = getattr(self.analyzer, "turns", [])
        for t in reversed(turns):
            if t.text.strip():
                return detect_lang(t.text)
        return self.default_lang

    def caller_spoke(self) -> None:
        self.strikes = 0

    def on_idle(self) -> SilenceAction:
        """Called when the timer fires. 1st: nudge. 2nd in a row: close + end."""
        self.strikes += 1
        lang = self.caller_lang()
        if self.strikes >= 2:
            return SilenceAction(CLOSE[lang], end_call=True)
        return SilenceAction(NUDGE[lang], end_call=False)
