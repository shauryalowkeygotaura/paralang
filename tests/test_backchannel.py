"""Backchannel vs real interruption."""
import asyncio

import pytest

pytest.importorskip("pipecat")
from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)

from paralang.adapters.pipecat import BackchannelAwareStartStrategy  # noqa: E402
from paralang.backchannel import is_backchannel  # noqa: E402


@pytest.mark.parametrize("text,bc", [
    ("haan", True), ("Haan ji.", True), ("hmm ok", True), ("हां जी", True), ("achha theek", True),
    ("haan but what is the price", False), ("no", False), ("ruko ek minute", False),
    ("haan ji haan ji haan", False),   # 5 tokens: that's talking, not listening
    ("हाँ।", True), ("हम्म", True), ("ठीक है", True), ("theek hai", True), ("hai kya?", False), ("अच्छा जी", True),
])
def test_is_backchannel(text, bc):
    assert is_backchannel(text) is bc


def run(frames_and_sleeps):
    async def go():
        s = BackchannelAwareStartStrategy(hold_s=0.15)
        started, resets = [], []
        s.add_event_handler("on_user_turn_started", lambda *a, **k: started.append(1))
        s.add_event_handler("on_reset_aggregation", lambda *a, **k: resets.append(1))
        for item in frames_and_sleeps:
            if isinstance(item, float):
                await asyncio.sleep(item)
            else:
                await s.process_frame(item)
        await asyncio.sleep(0.25)
        return s, started, resets
    return asyncio.run(go())


def tr(t):
    return TranscriptionFrame(text=t, user_id="u", timestamp="t")


def test_haan_while_bot_speaks_does_not_interrupt():
    s, started, resets = run([BotStartedSpeakingFrame(), VADUserStartedSpeakingFrame(), 0.05,
                              VADUserStoppedSpeakingFrame(), 0.3, tr("haan ji")])
    assert started == [] and resets == [1] and s.counts["backchannel"] == 1


def test_long_speech_while_bot_speaks_interrupts_after_hold():
    s, started, _ = run([BotStartedSpeakingFrame(), VADUserStartedSpeakingFrame(), 0.3])
    assert started == [1] and s.counts["interruption"] == 1


def test_short_but_real_words_interrupt():
    _, started, _ = run([BotStartedSpeakingFrame(), VADUserStartedSpeakingFrame(), 0.05,
                         VADUserStoppedSpeakingFrame(), 0.3, tr("nahi ruko")])
    assert started == [1]


def test_bot_silent_starts_immediately():
    _, started, _ = run([VADUserStartedSpeakingFrame()])
    assert started == [1]
