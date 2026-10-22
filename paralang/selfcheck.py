"""
selfcheck.py -- paralang listening to the AGENT's own voice (v0.3 #13a).

The same Analyzer that hears the caller, pointed at the agent's TTS output per
reply, compared with the caller it is talking to:

  too_fast   agent words/sec > FAST_RATIO x the caller's median pace
  monotone   pitch range < MONOTONE_ST semitones over a >= 3 s reply
  dead_air   the agent's reply started > DEAD_AIR_S after the caller stopped
             (from the Pipecat adapter's turn_timing)

Flags land in the call dump; paralang.tune counts them weekly and proposes
fixes (e.g. "lower the base speaking rate"). Framework-agnostic: feed it PCM.
"""
from __future__ import annotations

import statistics

from .features import extract

FAST_RATIO = 1.35
MONOTONE_ST = 2.0
MIN_REPLY_S = 3.0
DEAD_AIR_S = 2.5


class SelfCheck:
    def __init__(self, caller_analyzer=None):
        self.caller = caller_analyzer
        self.replies: list[dict] = []

    def _caller_pace(self) -> float | None:
        rates = [t.features.words_per_sec for t in getattr(self.caller, "turns", [])
                 if t.features.words_per_sec]
        return statistics.median(rates) if rates else None

    def observe_reply(self, pcm, sample_rate: int, text: str, reply_gap_s: float | None = None) -> dict:
        f = extract(pcm, sample_rate, text, acoustic="fast")
        flags = []
        pace = self._caller_pace()
        if pace and f.words_per_sec and f.words_per_sec > FAST_RATIO * pace:
            flags.append("too_fast")
        if f.voiced_s >= MIN_REPLY_S and f.f0_range_st is not None and f.f0_range_st < MONOTONE_ST:
            flags.append("monotone")
        if reply_gap_s is not None and reply_gap_s > DEAD_AIR_S:
            flags.append("dead_air")
        rec = {"reply": len(self.replies) + 1, "words_per_sec": f.words_per_sec,
               "caller_pace": round(pace, 2) if pace else None,
               "f0_range_st": round(f.f0_range_st, 2) if f.f0_range_st is not None else None,
               "voiced_s": round(f.voiced_s, 2), "reply_gap_s": reply_gap_s, "flags": flags}
        self.replies.append(rec)
        return rec
