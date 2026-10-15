"""Aura (Andromeda) client + fallbacks, no network."""
import io
import socket

import pytest

from paralang import aura, express
from paralang import say


class FakeResp(io.BytesIO):
    def __init__(self, data=b"", status=200, headers=None):
        super().__init__(data)
        self.status = status
        self._h = headers or {}

    def getheader(self, k, default=None):
        return self._h.get(k, default)


class FakeConn:
    def __init__(self):
        self.closed = False
        self.sock = None

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setattr(aura.time, "sleep", lambda s: None)
    aura._POOL.clear()


def test_raw_pcm_streamed_whole_samples_and_conn_reused(monkeypatch):
    pcm = bytes(range(256)) * 30 + b"\x07"
    seen = {}
    conn = FakeConn()

    def fake(path, body, headers, timeout):
        seen.update(path=path, timeout=timeout, headers=headers)
        return conn, FakeResp(pcm)
    monkeypatch.setattr(aura, "_request", fake)
    out = aura.synth("Namaste ji.", live=True)
    assert out == pcm[:-1]
    assert "model=aura-2-andromeda-en" in seen["path"] and "container=none" in seen["path"]
    assert seen["timeout"] == aura.LIVE_FIRST_BYTE_S and seen["headers"]["Connection"] == "keep-alive"
    assert aura._POOL == [conn] and not conn.closed          # fully read -> back in the pool


def test_abandoned_stream_closes_connection(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(aura, "_request", lambda *a: (conn, FakeResp(b"\x00\x01" * 20000)))
    g = aura.stream("Hello there.")
    next(g)
    g.close()                                    # caller barged in mid-sentence
    assert conn.closed and aura._POOL == []


def test_live_slow_fails_fast(monkeypatch):
    calls = {"n": 0}

    def slow(*a):
        calls["n"] += 1
        raise socket.timeout("timed out")
    monkeypatch.setattr(aura, "_request", slow)
    with pytest.raises(aura.AuraError):
        aura.synth("Hello.", live=True)
    assert calls["n"] == 1                       # live: no retries, caller falls back


def test_batch_retries_timeouts_and_5xx(monkeypatch):
    calls = {"n": 0}

    def flaky(*a):
        calls["n"] += 1
        if calls["n"] == 1:
            raise socket.timeout("timed out")
        if calls["n"] == 2:
            return FakeConn(), FakeResp(b"busy", status=503)
        return FakeConn(), FakeResp(b"\x01\x00" * 10)
    monkeypatch.setattr(aura, "_request", flaky)
    assert aura.synth("Hello.") and calls["n"] == 3


def test_stale_keepalive_retried_on_fresh_connection(monkeypatch):
    import http.client

    class Stale(FakeConn):
        def request(self, *a, **k):
            raise http.client.RemoteDisconnected("server closed idle conn")

    class Fresh(FakeConn):
        def request(self, *a, **k):
            pass

        def getresponse(self):
            return FakeResp(b"\x02\x00" * 10)
    aura._POOL.append(Stale())
    monkeypatch.setattr(aura, "_new_conn", lambda timeout: Fresh())
    assert aura.synth("Hi.", live=True) == b"\x02\x00" * 10   # even live: stale fails instantly


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
    s = express("Hello there, good morning. आपका प्लान कल खत्म होगा। This will fail now, sadly enough.", wobble=False)
    say.aura_pcm(s)
    assert used == [("aura", "aura-2-andromeda-en"), ("edge", "hi-IN-SwaraNeural"),
                    ("edge", "en-IN-NeerjaNeural")]
