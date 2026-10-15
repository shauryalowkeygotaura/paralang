"""Orpheus client logic, no network: key rotation, live vs batch rate limits,
WAV header stripping, script routing, alignment pairing."""
import io
import json
import struct
import urllib.error

import pytest

from paralang import express, orpheus
from paralang import say


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for n in ("GROQ_API_KEYS", "GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("GROQ_API_KEYS", "k1,k2")
    orpheus._Pool.cooling.clear()
    orpheus._Pool.turn = 0


def streaming_wav(pcm: bytes) -> bytes:
    # Groq streams WAV with a placeholder size, like this
    return (b"RIFF\xff\xff\xff\xffWAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
            + b"data\xff\xff\xff\xff" + pcm)


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_429():
    return urllib.error.HTTPError("u", 429, "rate", {"retry-after": "30"}, io.BytesIO(b"slow down"))


def test_speakable_routes_scripts():
    assert orpheus.speakable("[warm] Haan ji, aapka plan kal expire hoga.")   # romanised Hinglish
    assert not orpheus.speakable("नमस्ते, आपका प्लान")                             # Devanagari
    assert not orpheus.speakable("   ")


def test_stream_strips_header_and_keeps_samples_whole(monkeypatch):
    pcm = bytes(range(256)) * 41 + b"\x01"          # odd length: last byte must not leak
    monkeypatch.setattr(orpheus.urllib.request, "urlopen", lambda req, timeout: Resp(streaming_wav(pcm)))
    out = orpheus.synth("Hi there.", "hannah")
    assert out == pcm[: len(pcm) - 1] and len(out) % 2 == 0


def test_429_rotates_to_next_key(monkeypatch):
    used = []

    def fake(req, timeout):
        key = req.headers["Authorization"].split()[1]
        used.append(key)
        if key == "k1":
            raise http_429()
        return Resp(streaming_wav(b"\x00\x01" * 10))
    monkeypatch.setattr(orpheus.urllib.request, "urlopen", fake)
    assert orpheus.synth("Hi.", live=True)
    assert used == ["k1", "k2"]
    # k1 is cooling now: the next call goes straight to k2
    used.clear()
    orpheus.synth("Again.", live=True)
    assert used == ["k2"]


def test_live_mode_never_waits(monkeypatch):
    monkeypatch.setattr(orpheus.urllib.request, "urlopen", lambda req, timeout: (_ for _ in ()).throw(http_429()))
    with pytest.raises(orpheus.RateLimited):
        orpheus.synth("Hi.", live=True)


def test_connection_reset_retried_in_batch_failfast_live(monkeypatch):
    calls = {"n": 0}

    def flaky(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError(10054, "forcibly closed")
        return Resp(streaming_wav(b"\x00\x01" * 10))
    monkeypatch.setattr(orpheus.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(orpheus.time, "sleep", lambda s: None)
    assert orpheus.synth("Hi.")                      # batch: retried, succeeded
    calls["n"] = 0
    with pytest.raises(orpheus.OrpheusError):
        orpheus.synth("Hi.", live=True)              # live: fail fast, caller falls back


def test_bad_voice_and_no_key(monkeypatch):
    with pytest.raises(ValueError):
        orpheus.synth("Hi.", voice="neerja")
    monkeypatch.delenv("GROQ_API_KEYS")
    with pytest.raises(orpheus.OrpheusError):
        orpheus.synth("Hi.")


def test_orpheus_pcm_sends_hindi_to_edge_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(orpheus, "synth", lambda text, voice, live=False: calls.append(("orpheus", text)) or b"OR")
    monkeypatch.setattr(say, "_edge_lines", lambda lines, voice=None: calls.append(("edge", lines[0].text)) or b"ED")
    s = express("[warm] Namaste ji. आपका प्लान कल खत्म होगा। [calm] Koi jaldi nahi.", wobble=False)
    say.orpheus_pcm(s, "hannah")
    assert [c[0] for c in calls] == ["orpheus", "edge", "orpheus"]
    assert calls[0][1].startswith("[warm]") and calls[2][1].startswith("[calm]")


def test_rate_limited_chunk_falls_back_not_dropped(monkeypatch):
    def limited(*a, **k):
        raise orpheus.RateLimited("all cooling")
    spoken = []

    async def fake_edge(text, voice, kw):
        spoken.append(text)
        return b"ED"
    monkeypatch.setattr(orpheus, "synth", limited)
    monkeypatch.setattr(say, "_synth", fake_edge)
    pcm = say.orpheus_pcm(express("[excited] Big news today!"), "troy", live=True)
    assert pcm.endswith(b"ED") and spoken == ["Big news today!"]   # direction tag stripped for edge


def test_default_engine_is_orpheus():
    assert say.DEFAULT_ENGINE == "orpheus"


def test_long_audio_aligned_in_windows_cut_at_silence(monkeypatch):
    import array
    sr = 1000                                   # tiny rate keeps the test fast
    monkeypatch.setattr(orpheus, "ALIGN_WINDOW_S", 10)
    monkeypatch.setattr(orpheus, "ALIGN_SEEK_S", 3)
    loud = array.array("h", [9000, -9000] * 500).tobytes()      # 1 s of "speech"
    quiet = b"\x00\x00" * 200                                    # 0.2 s gap
    pcm = (loud * 8 + quiet + loud * 6)                          # gap sits at ~8.0-8.2 s
    windows = []

    def fake_once(chunk, rate):
        windows.append(len(chunk) / 2 / rate)
        return [("w", 0.5, 0.9)]
    monkeypatch.setattr(orpheus, "_align_once", fake_once)
    words = orpheus.align(pcm, "", sr)
    assert len(windows) == 2 and 7.9 <= windows[0] <= 8.25     # cut inside the silence, not at 10 s
    assert words[1][1] == pytest.approx(windows[0] + 0.5, abs=0.01)  # second window offset back


def test_align_pairs_submitted_words_with_whisper_times(monkeypatch):
    whisper = {"words": [{"word": " Hi", "start": 0.1, "end": 0.3}, {"word": " Ria.", "start": 0.3, "end": 0.8}]}
    monkeypatch.setattr(orpheus.urllib.request, "urlopen",
                        lambda req, timeout: Resp(json.dumps(whisper).encode()))
    assert orpheus.align(b"\x00\x00" * 100, "[warm] Hi Riya.") == [("Hi", 0.1, 0.3), ("Riya.", 0.3, 0.8)]
    # word counts differ: trust Whisper's words rather than mis-time the script
    assert orpheus.align(b"\x00\x00" * 100, "Hi there Riya.")[1][0] == "Ria."
