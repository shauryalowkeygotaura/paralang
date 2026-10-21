"""Drive ParalangProcessor with the frame sequence a real call produces
(pipecat's own run_test harness): bot speaks -> caller waits 2.6 s -> speaks
-> transcript. The transcript must come out tagged and the turn recorded."""
import asyncio
import time

import numpy as np
import pytest

pipecat = pytest.importorskip("pipecat")
from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InputAudioRawFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test  # noqa: E402

from paralang import Analyzer  # noqa: E402
from paralang.adapters.pipecat import ParalangProcessor  # noqa: E402

SR = 16000


def voice_chunks(secs, chunk_s=0.02):
    t = np.arange(int(secs * SR)) / SR
    x = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 8))
    pcm = (0.3 * x / np.abs(x).max() * 32767).astype(np.int16).tobytes()
    n = int(chunk_s * SR) * 2
    return [InputAudioRawFrame(audio=pcm[i:i + n], sample_rate=SR, num_channels=1)
            for i in range(0, len(pcm), n)]


@pytest.mark.asyncio
async def test_transcript_gets_tagged_after_slow_answer():
    turns = []
    proc = ParalangProcessor(Analyzer(), on_turn=turns.append)
    t_bot_stop = time.time()
    frames = [
        BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(),
        # VAD confirmed 2.6 s after the bot stopped (timestamp is wall-clock)
        VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=t_bot_stop + 2.8),
        *voice_chunks(1.0),
        VADUserStoppedSpeakingFrame(stop_secs=0.8),
        SleepFrame(sleep=0.3),
        TranscriptionFrame(text="haan theek hai", user_id="u", timestamp="t"),
    ]
    down, _up = await run_test(proc, frames_to_send=frames, expected_down_frames=None)
    texts = [f.text for f in down if isinstance(f, TranscriptionFrame)]
    assert texts and texts[0].startswith("[tone:") and texts[0].endswith("haan theek hai"), texts
    assert len(turns) == 1
    f = turns[0].features
    assert 2.3 < f.response_latency_s < 3.0, f.response_latency_s
    assert f.words == 3 and f.f0_median_st is not None
    # audio frames must still flow downstream untouched
    assert sum(isinstance(f, InputAudioRawFrame) for f in down) == 50


@pytest.mark.asyncio
async def test_quick_answer_passes_through_untagged():
    proc = ParalangProcessor(Analyzer())
    t = time.time()
    frames = [
        BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(),
        VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=t + 0.6),
        *voice_chunks(1.0),
        VADUserStoppedSpeakingFrame(stop_secs=0.8),
        SleepFrame(sleep=0.3),
        TranscriptionFrame(text="haan ji", user_id="u", timestamp="t"),
    ]
    down, _ = await run_test(proc, frames_to_send=frames, expected_down_frames=None)
    assert [f.text for f in down if isinstance(f, TranscriptionFrame)] == ["haan ji"]


@pytest.mark.asyncio
async def test_language_hint_emitted_on_switch():
    hints = []
    proc = ParalangProcessor(Analyzer(acoustic=False), on_language=hints.append)
    frames = []
    for text in ("Mera plan kab khatam hoga ji", "Haan ji bataiye kitne ka hai"):
        t = time.time()
        frames += [VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=t), *voice_chunks(0.6),
                   VADUserStoppedSpeakingFrame(stop_secs=0.8), SleepFrame(sleep=0.2),
                   TranscriptionFrame(text=text, user_id="u", timestamp="t")]
    await run_test(proc, frames_to_send=frames, expected_down_frames=None)
    assert hints == ["hi"]


def test_auto_language_stt_sends_no_language_until_hinted():
    from paralang.adapters.pipecat import AutoLanguageGroqSTT
    sent = []

    class FakeClient:
        class audio:
            class transcriptions:
                @staticmethod
                async def create(**kw):
                    sent.append(kw)
    stt = AutoLanguageGroqSTT(api_key="x", settings=AutoLanguageGroqSTT.Settings(model="whisper-large-v3-turbo"))
    stt._client = FakeClient()
    asyncio.run(stt._transcribe(b"RIFF"))
    stt.set_language_hint("hi")
    asyncio.run(stt._transcribe(b"RIFF"))
    assert "language" not in sent[0] and sent[1]["language"] == "hi"


@pytest.mark.asyncio
async def test_turn_timing_reply_gap_and_cut_off():
    proc = ParalangProcessor(Analyzer(acoustic=False))
    t = time.time()
    frames = [
        VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=t), *voice_chunks(0.6),
        VADUserStoppedSpeakingFrame(stop_secs=0.2), SleepFrame(sleep=0.2),
        TranscriptionFrame(text="mera plan kab", user_id="u", timestamp="t"),
        SleepFrame(sleep=0.3), BotStartedSpeakingFrame(),             # reply after ~0.5 s
        SleepFrame(sleep=0.2),
        VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=time.time() + 0.7),  # caller resumes: cut off
    ]
    await run_test(proc, frames_to_send=frames, expected_down_frames=None)
    gaps = proc.turn_timing["reply_gaps"]
    assert len(gaps) == 1 and 0.3 < gaps[0] < 1.5
    assert proc.turn_timing["cut_offs"] == 1
    assert proc.analyzer.turns[-1].features.extras["cut_off"] is True
