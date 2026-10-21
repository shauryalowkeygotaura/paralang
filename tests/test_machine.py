"""Answering machine / IVR / hold-music detection."""
import time

import numpy as np
import pytest

from paralang import Analyzer
from paralang.features import TurnFeatures
from paralang.machine import beep_seconds, classify

SR = 16000


def tone(hz, secs, amp=0.3):
    t = np.arange(int(secs * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def speechy(secs):
    t = np.arange(int(secs * SR)) / SR
    x = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 10))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def feats(**kw):
    base = dict(duration_s=2, voiced_s=1.0, words=2, words_per_sec=2, pause_count=0,
                pause_total_s=0, longest_pause_s=0, pause_ratio=0, fillers=0)
    base.update(kw)
    return TurnFeatures(**base)


def test_beep_found_and_speech_is_not_a_beep():
    x = np.concatenate([np.zeros(SR // 2, np.float32), tone(1000, 0.5), np.zeros(SR // 4, np.float32)])
    assert 0.4 <= beep_seconds(x, SR) <= 0.55
    assert beep_seconds(speechy(2.0), SR) < 0.1          # harmonic speech spreads its energy
    assert beep_seconds(tone(300, 1.0), SR) == 0.0       # outside the beep band


@pytest.mark.parametrize("text,kind", [
    ("Please leave a message after the tone", "voicemail"),
    ("Aap jis number par call kar rahe hain woh abhi uplabdh nahi hai", "voicemail"),
    ("Press 1 for English, Hindi ke liye 2 dabaye", "ivr"),
    ("Hello?", "human"),
])
def test_phrases(text, kind):
    assert classify(feats(words=len(text.split())), text, 1).kind == kind


def test_long_greeting_is_machine_short_hello_is_human():
    assert classify(feats(voiced_s=4.2, words=12), "x " * 12, 1).kind == "voicemail"
    assert classify(feats(voiced_s=0.6, words=1), "haan ji", 1).kind == "human"
    assert classify(feats(voiced_s=4.2, words=12), "x " * 12, 3).kind == "human"   # only turns 1-2


def test_analyzer_turn1_cue_and_tag():
    an = Analyzer(acoustic=False)
    res = an.analyze_turn(speechy(1.0), SR, "The number you are calling is not reachable, please try later")
    assert any(c.name == "machine" and c.evidence["kind"] == "voicemail" for c in res.cues)
    assert "answering machine" in res.tag


pipecat = pytest.importorskip("pipecat")


@pytest.mark.asyncio
async def test_live_beep_outside_vad_fires_on_machine():
    from pipecat.frames.frames import InputAudioRawFrame
    from pipecat.tests.utils import run_test
    from paralang.adapters.pipecat import ParalangProcessor
    got = []
    proc = ParalangProcessor(Analyzer(acoustic=False), on_machine=got.append)
    audio = np.concatenate([np.zeros(SR // 2, np.float32), tone(1000, 0.6), np.zeros(SR // 2, np.float32)])
    pcm = (audio * 32767).astype(np.int16).tobytes()
    n = int(0.02 * SR) * 2
    frames = [InputAudioRawFrame(audio=pcm[i:i + n], sample_rate=SR, num_channels=1)
              for i in range(0, len(pcm), n)]
    await run_test(proc, frames_to_send=frames, expected_down_frames=None)
    assert got and got[0].kind == "voicemail"
