"""Fish Audio engine: payload, cue rendering, fallback. No network."""
import io
import json

import pytest

from paralang import fish, say
from paralang.emotion import express
from paralang.tts import fish_chunks, to_fish


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_cues_beats_emphasis_and_directions():
    s = express("[warm] Hi, this is Riya. [empathetic:0.5] So sorry. [laughing] Ha ha. "
                "[dramatic] And none of them | are *theoretical*.")
    (text,) = to_fish(s)
    assert text.startswith("[warm and friendly] Hi, this is Riya.")
    assert "[slightly empathetic][soft tone] So sorry." in text
    assert "[laughing] Ha ha." in text                          # Fish acts it, kept
    assert "[dramatic] And none of them [break] are [emphasis] theoretical." in text


def test_neutral_has_no_cue_and_chunks_restate_cue():
    assert to_fish(express("Plain words here.")) == ["Plain words here."]
    s = express("[sad] " + " ".join(["This is a long sad sentence."] * 60))
    chunks = to_fish(s, max_chars=300)
    assert len(chunks) > 1 and all(c.startswith("[sad] ") for c in chunks)
    assert sum(len(lines) for _, lines in fish_chunks(s, 300)) == len(s.lines)


def test_payload(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["body"] = json.loads(req.data)
        return _Resp(b"\x01\x00" * 100)

    monkeypatch.setenv("FISH_API_KEY", "k")
    monkeypatch.delenv("FISH_MODEL", raising=False)
    monkeypatch.setenv("FISH_VOICE_ID", "abcdef1234567890")
    monkeypatch.setattr(fish.urllib.request, "urlopen", fake_urlopen)
    assert len(fish.synth("[happy] Hi.")) == 200
    assert seen["headers"]["model"] == "s2.1-pro-free"
    assert seen["headers"]["authorization"] == "Bearer k"
    b = seen["body"]
    assert b["format"] == "pcm" and b["sample_rate"] == 24000
    assert b["reference_id"] == "abcdef1234567890" and "prosody" not in b


def test_bad_model_and_voice_rejected(monkeypatch):
    with pytest.raises(ValueError):
        fish.resolve_model("s1")
    with pytest.raises(ValueError):
        fish.resolve_voice("bad id; rm -rf")


def test_missing_key_is_loud(monkeypatch):
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    with pytest.raises(fish.FishError, match="FISH_API_KEY"):
        fish.synth("Hi.")


def test_fish_failure_falls_back_to_aura(monkeypatch):
    calls = []

    def boom(text, voice=None, **kw):
        raise fish.FishError("down")

    monkeypatch.setattr(fish, "synth", boom)
    monkeypatch.setattr(say, "aura_pcm", lambda script, **kw: calls.append(
        [l.text for l in script.lines]) or b"\x00\x00" * 10)
    say.fish_pcm(express("[warm] One. Two."))
    assert calls == [["One.", "Two."]]
