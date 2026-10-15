"""Aura (Andromeda) client + fallbacks, no network."""
import io
import socket
import urllib.error

import pytest

from paralang import aura, express
from paralang import say


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setattr(aura.time, "sleep", lambda s: None)


def test_raw_pcm_streamed_whole_samples(monkeypatch):
    pcm = bytes(range(256)) * 30 + b"\x07"
    seen = {}

    def fake(req, timeout):
        seen["url"], seen["timeout"] = req.full_url, timeout
        return Resp(pcm)
    monkeypatch.setattr(aura.urllib.request, "urlopen", fake)
    out = aura.synth("Namaste ji.", live=True)
    assert out == pcm[:-1]
    assert "model=aura-2-andromeda-en" in seen["url"] and "container=none" in seen["url"]
    assert seen["timeout"] == aura.LIVE_FIRST_BYTE_S


def test_live_slow_first_byte_fails_fast(monkeypatch):
    calls = {"n": 0}

    def slow(req, timeout):
        calls["n"] += 1
        raise socket.timeout("timed out")
    monkeypatch.setattr(aura.urllib.request, "urlopen", slow)
    with pytest.raises(aura.AuraError):
        aura.synth("Hello.", live=True)
    assert calls["n"] == 1                       # live: no retries, caller falls back


def test_batch_retries_timeouts_and_5xx(monkeypatch):
    calls = {"n": 0}

    def flaky(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise socket.timeout("timed out")
        if calls["n"] == 2:
            raise urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b""))
        return Resp(b"\x01\x00" * 10)
    monkeypatch.setattr(aura.urllib.request, "urlopen", flaky)
    assert aura.synth("Hello.") and calls["n"] == 3


def test_voice_whitelist_and_key(monkeypatch):
    with pytest.raises(ValueError):
        aura.synth("Hi.", voice="andromeda&x=1")
    monkeypatch.delenv("DEEPGRAM_API_KEY")
    with pytest.raises(aura.AuraError):
        aura.synth("Hi.")


def test_aura_pcm_fallbacks_neerja_and_swara(monkeypatch):
    used = []

    def fake_aura(text, voice, live=False):
        if "fail" in text:
            raise aura.AuraError("boom")
        used.append(("aura", voice))
        return b"AU"

    async def fake_edge(text, voice, kw):
        used.append(("edge", voice))
        return b"ED"
    monkeypatch.setattr(aura, "synth", fake_aura)
    monkeypatch.setattr(say, "_synth", fake_edge)
    s = express("Namaste ji. आपका प्लान कल खत्म होगा। This will fail now.", wobble=False)
    say.aura_pcm(s)
    assert used == [("aura", "aura-2-andromeda-en"), ("edge", "hi-IN-SwaraNeural"),
                    ("edge", "en-IN-NeerjaNeural")]
