"""Hybrid mode: fast critical path, nothing dropped."""
import time

import numpy as np
import pytest

from paralang import Analyzer

pytest.importorskip("opensmile")
SR = 16000


def voice(secs, f0=150):
    t = np.arange(int(secs * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def test_hybrid_is_default_and_critical_path_is_fast():
    an = Analyzer()
    assert an.acoustic == "hybrid"
    # one-time worker start + openSMILE load (~12 s, longer on a busy box); the
    # 30 ms budget below is for steady-state turns, which is what a call sees
    an._pool.submit(lambda: None).result(120)
    # Measured 2026-10-22: 9-19 ms in-suite, 0 GC collections during timing; the
    # rare >30 ms runs coincide with whole-machine load. One re-measure separates
    # an OS scheduling spike from a real regression (which fails both rounds).
    for attempt in range(2):
        times = []
        for _ in range(7):
            t0 = time.perf_counter()
            res = an.analyze_turn(voice(3.0), SR, "haan theek hai")
            times.append((time.perf_counter() - t0) * 1000)
        if np.median(times) < 30:
            break
    assert np.median(times) < 30, times          # the 0.03 s budget
    assert res.features.f0_median_st is not None  # cues have pitch immediately
    an.close()


def test_hybrid_keeps_voice_quality_after_enrichment():
    an = Analyzer()
    an.analyze_turn(voice(2.0), SR, "haan ji")
    rows = an.export(timeout=60)                 # waits for prewarm + the background pass
    f = rows[0]["features"]
    assert f["jitter"] is not None and f["shimmer_db"] is not None and f["hnr_db"] is not None
    smile_hz = 27.5 * 2 ** (f["extras"]["smile_f0_median_st"] / 12)
    fast_hz = 27.5 * 2 ** (f["f0_median_st"] / 12)
    assert abs(smile_hz - fast_hz) / smile_hz < 0.05   # the two trackers agree
    an.close()


def test_unvoiced_turn_queues_nothing():
    an = Analyzer()
    an.analyze_turn(np.zeros(SR, dtype=np.float32), SR, "")
    assert an._enrich == []
    an.close()
