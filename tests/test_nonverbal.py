"""Laugh / sigh / breath heuristics on synthetic signals."""
import numpy as np

from paralang import Analyzer
from paralang.features import _frame_db, pitch_track
from paralang.nonverbal import detect, merge_words

SR = 16000
rng = np.random.default_rng(7)


def voiced(secs, f0=180):
    t = np.arange(int(secs * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def breath(secs, amp):
    return (rng.normal(0, amp, int(secs * SR))).astype(np.float32)


def quiet(secs):
    return breath(secs, 1e-4)


def run(x, text=""):
    return detect(_frame_db(x, SR), pitch_track(x, SR), text)


def test_regular_breathy_bursts_are_a_laugh():
    # "ha-ha-ha-ha-ha": 5 bursts, 0.2 s apart (5/s), breathy unvoiced gaps
    burst = np.concatenate([voiced(0.07, 260), breath(0.13, 0.03)])
    x = np.concatenate([quiet(0.4), np.tile(burst, 5), quiet(0.4)])
    events, _ = run(x)
    assert [e.kind for e in events] == ["laugh"] and events[0].via == "acoustic"


def test_ordinary_speech_rhythm_is_not_a_laugh():
    # irregular syllables (speech) with voiced, not breathy, gaps
    parts, r = [quiet(0.3)], np.random.default_rng(1)
    for _ in range(10):
        parts.append(voiced(r.uniform(0.08, 0.25), 150))
        parts.append(voiced(r.uniform(0.03, 0.12), 120) * 0.3)
    events, _ = run(np.concatenate(parts + [quiet(0.3)]))
    assert not any(e.kind == "laugh" for e in events)


def test_fading_exhale_is_a_sigh():
    t = np.linspace(0, 1, int(0.9 * SR))
    exhale = breath(0.9, 0.02) * (1 - 0.85 * t).astype(np.float32)
    x = np.concatenate([voiced(0.6), quiet(0.2), exhale, quiet(0.4)])
    events, _ = run(x)
    assert any(e.kind == "sigh" for e in events)


def test_words_add_and_boost():
    assert merge_words([], "hahaha theek hai")[0]["kind"] == "laugh"
    boosted = merge_words([{"kind": "sigh", "start_s": 1, "end_s": 2, "strength": 0.5, "via": "acoustic"}], "uff")
    assert boosted[0]["via"] == "both" and boosted[0]["strength"] == 0.8


def test_cues_and_tag():
    res = Analyzer(acoustic="fast").analyze_turn(np.concatenate([voiced(0.8), quiet(0.5)]), SR, "hahaha haan ji")
    assert any(c.name == "laughed" for c in res.cues) and "laughed" in res.tag
