"""Synthetic-signal tests: every feature checked against a signal whose true
value we built by hand. If these pass, the measurements are right; whether
the CUES are right can only be learned from real calls."""
import json
import math
import wave

import numpy as np
import pytest

from paralang import Analyzer, extract
from paralang.adapters.offline import analyze_file
from paralang.baseline import Baseline

SR = 16000


def tone(f0_hz, secs, amp=0.3, sr=SR, glide_to=None):
    """Harmonic-rich 'voice' (opensmile's pitch tracker wants harmonics, not a pure sine)."""
    t = np.arange(int(secs * sr)) / sr
    f = np.full_like(t, f0_hz) if glide_to is None else np.geomspace(f0_hz, glide_to, t.size)
    phase = 2 * np.pi * np.cumsum(f) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 8))
    return (amp * x / np.abs(x).max()).astype(np.float32)


def silence(secs, sr=SR):
    return (np.random.default_rng(0).normal(0, 1e-4, int(secs * sr))).astype(np.float32)


def st(hz):
    return 12 * math.log2(hz / 27.5)


def test_pitch_within_5_percent():
    f = extract(tone(150, 2.0), SR)
    hz = 27.5 * 2 ** (f.f0_median_st / 12)
    assert abs(hz - 150) / 150 < 0.05, hz
    assert f.f0_range_st < 1.0          # steady tone = no liveliness


def test_rising_end_detected():
    x = np.concatenate([tone(140, 1.2), tone(140, 0.5, glide_to=220)])
    f = extract(x, SR)
    assert f.f0_final_slope_st > 6, f.f0_final_slope_st


def test_pauses_measured_within_50ms():
    x = np.concatenate([tone(150, 0.8), silence(1.2), tone(150, 0.6), silence(0.4), tone(150, 0.5)])
    f = extract(x, SR, acoustic=False)
    assert f.pause_count == 2
    assert abs(f.longest_pause_s - 1.2) < 0.05
    assert abs(f.pause_total_s - 1.6) < 0.08


def test_leading_and_trailing_silence_are_not_pauses():
    x = np.concatenate([silence(1.0), tone(150, 1.0), silence(1.0)])
    f = extract(x, SR, acoustic=False)
    assert f.pause_count == 0 and f.longest_pause_s == 0.0


def test_short_gaps_are_articulation_not_pauses():
    x = np.concatenate([tone(150, 0.5), silence(0.12), tone(150, 0.5)])
    assert extract(x, SR, acoustic=False).pause_count == 0


def test_rate_and_fillers_from_text():
    f = extract(tone(150, 2.0), SR, "umm matlab haan theek hai, uh", acoustic=False)
    assert f.words == 6 and f.fillers == 3
    assert abs(f.words_per_sec - 3.0) < 0.2


def test_8khz_phone_audio_and_bytes_input():
    pcm = (tone(150, 1.5, sr=8000) * 32767).astype(np.int16).tobytes()
    f = extract(pcm, 8000)
    assert abs(27.5 * 2 ** (f.f0_median_st / 12) - 150) / 150 < 0.08


def test_too_little_voice_skips_acoustics():
    f = extract(np.concatenate([silence(1.0), tone(150, 0.2)]), SR)
    assert f.f0_median_st is None and f.words_per_sec is None


def test_baseline_z_and_freeze():
    from paralang.features import TurnFeatures
    def tf(rate):
        return TurnFeatures(1, 1, 3, rate, 0, 0, 0, 0, 0)
    b = Baseline(warmup=3)
    assert b.z(tf(3.0)) == {}                    # no history = unknown, not zero
    for r in (2.8, 3.0, 3.2):
        b.update(tf(r))
    assert b.frozen
    assert b.z(tf(5.0))["words_per_sec"] > 3     # much faster than this caller's normal
    b.update(tf(9.0))                            # frozen: drift must not move the baseline
    assert b.z(tf(5.0))["words_per_sec"] > 3


def test_long_think_and_tag_and_json_contract():
    an = Analyzer()
    res = an.analyze_turn(tone(150, 1.0), SR, "haan", response_latency_s=3.2)
    assert any(c.name == "long_think" for c in res.cues)
    assert res.tag.startswith("[tone:") and "long pause" in res.tag
    assert res.annotate().endswith("haan")
    d = json.loads(json.dumps(res.to_dict()))   # must be plain-JSON serialisable
    assert d["schema"] == "1" and d["turn"] == 1 and d["features"]["response_latency_s"] == 3.2
    assert "tone: long_think" in an.call_signals()


def test_no_cues_means_untouched_text():
    an = Analyzer()
    res = an.analyze_turn(tone(150, 1.0), SR, "haan theek hai", response_latency_s=0.4)
    assert res.tag == "" and res.annotate() == "haan theek hai"


def test_offline_stereo_channel_and_turns(tmp_path):
    caller = np.concatenate([silence(1.0), tone(150, 1.0), silence(1.0)])
    agent = np.concatenate([tone(220, 1.0), silence(2.0)])
    stereo = (np.stack([agent, caller], axis=1) * 32767).astype(np.int16)
    p = tmp_path / "call.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(stereo.tobytes())
    rows = analyze_file(p, [{"start": 0.9, "end": 2.1, "text": "haan ji",
                             "response_latency_s": None}], channel=1)
    assert len(rows) == 1
    hz = 27.5 * 2 ** (rows[0]["features"]["f0_median_st"] / 12)
    assert abs(hz - 150) / 150 < 0.05            # picked the caller, not the 220 Hz agent


def test_extraction_is_fast_enough():
    an = Analyzer()
    res = an.analyze_turn(tone(150, 5.0), SR, "x")
    assert res.elapsed_ms < 600, res.elapsed_ms   # runs off the event loop, in parallel with STT


def test_bad_format_rejected():
    with pytest.raises(ValueError):
        extract(b"\x00\x00" * 100, 0)


@pytest.mark.parametrize("hz", [60, 70, 85, 395])
def test_deep_and_high_voices_tracked(hz):
    # before the Boersma window correction, <75 Hz (deep male) never voiced
    from paralang.features import pitch_track
    f = pitch_track(tone(hz, 2.0), SR)
    v = f[f > 0]
    assert len(v) > 0.9 * len(f)
    assert abs(np.median(v) - hz) / hz < 0.02


def test_noise_is_not_voiced():
    from paralang.features import pitch_track
    noise = np.random.default_rng(1).normal(0, 0.05, SR * 2).astype(np.float32)
    assert (pitch_track(noise, SR) > 0).sum() == 0
