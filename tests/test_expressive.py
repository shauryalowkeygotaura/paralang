"""assemble (one take from many clips), perform (emotion as punctuation),
and the expressive / flow switches in say.py. No network."""
import numpy as np

from paralang import assemble, say
from paralang.assemble import SR, Clip
from paralang.emotion import express
from paralang.perform import shape


def _tone_pcm(seconds, db=-20.0, lead=0.3, tail=0.3, hz=180.0):
    t = np.arange(int(SR * seconds)) / SR
    x = np.sin(2 * np.pi * hz * t) * 10 ** (db / 20) * np.sqrt(2)
    x = np.concatenate([np.zeros(int(SR * lead)), x, np.zeros(int(SR * tail))])
    return (x * 32767).astype("<i2").tobytes()


def _f(pcm):
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768


def test_trim_cuts_engine_padding_but_keeps_margins():
    x = _f(_tone_pcm(1.0, lead=0.4, tail=0.5))
    y = assemble.trim(x)
    assert abs(len(y) / SR - (1.0 + assemble.LEAD_KEEP_S + assemble.TAIL_KEEP_S)) < 0.03


def test_join_gap_is_exactly_the_planner_gap():
    a, b = _tone_pcm(0.5, lead=0.3, tail=0.4), _tone_pcm(0.5, lead=0.3, tail=0.4)
    out = _f(assemble.join([Clip(a), Clip(b, pause_before_ms=300)]))
    expected = 2 * (0.5 + assemble.LEAD_KEEP_S + assemble.TAIL_KEEP_S) + 0.3
    assert abs(len(out) / SR - expected) < 0.05
    # raw (old behaviour) keeps all the padding
    assert len(_f(assemble.raw([Clip(a), Clip(b, 300)]))) / SR > expected + 0.6


def test_join_levels_clips_and_applies_emotion_gain():
    quiet, loud = _tone_pcm(0.6, db=-28), _tone_pcm(0.6, db=-12)
    out = _f(assemble.join([Clip(quiet), Clip(loud, 200)], tone=False))
    half = len(out) // 2
    l1, l2 = assemble.speech_dbfs(out[:half]), assemble.speech_dbfs(out[half:])
    assert abs(l1 - l2) < 1.5 and abs(l1 - assemble.TARGET_SPEECH_DBFS) < 1.5
    # gain is capped: a near-silent clip is not pumped up into noise
    capped = _f(assemble.join([Clip(_tone_pcm(0.6, db=-40))], tone=False))
    assert abs(assemble.speech_dbfs(capped) - (-40 + assemble.MAX_GAIN_DB)) < 1.5
    soft = _f(assemble.join([Clip(loud, gain_db=-6)], tone=False))
    assert abs(assemble.speech_dbfs(soft) - (assemble.TARGET_SPEECH_DBFS - 6)) < 1.5


def test_gaps_are_room_tone_not_digital_zero():
    out = _f(assemble.join([Clip(_tone_pcm(0.3)), Clip(_tone_pcm(0.3), 400)]))
    mid = len(out) // 2
    gap = out[mid - 1000: mid + 1000]
    rms_db = 20 * np.log10(np.sqrt((gap ** 2).mean()) + 1e-12)
    assert -70 < rms_db < -55


def test_join_never_clips():
    hot = (np.ones(SR // 2) * 0.99 * 32767).astype("<i2").tobytes()
    out = _f(assemble.join([Clip(hot, gain_db=9)]))
    assert np.abs(out).max() <= assemble.PEAK_CEILING + 1e-3


def test_volume_db():
    assert assemble.volume_db(0) == 0
    assert assemble.volume_db(-20) < -1.5


def test_shape_endings():
    assert shape("Your appointment is confirmed.", "excited") == "Your appointment is confirmed!"
    assert shape("Your appointment is confirmed.", "excited", 0.3) == "Your appointment is confirmed."
    assert shape("Guess what?", "excited") == "Guess what?"            # questions untouched
    assert shape("It was gone.", "sad") == "It was gone..."
    assert shape("Wait for it...", "excited") == "Wait for it..."      # author's trail kept
    assert shape("Take care.", "warm") == "Take care."                 # no rule -> unchanged
    assert shape("No punctuation", "urgent") == "No punctuation!"


def _fake_aura(calls):
    def synth(text, voice, live=False):
        calls.append(text)
        return _tone_pcm(0.2, lead=0.1, tail=0.1)
    return synth


def test_aura_one_request_per_sentence_unless_flow(monkeypatch):
    from paralang import aura
    calls = []
    monkeypatch.setattr(aura, "synth", _fake_aura(calls))
    s = express("[serious] One thing. Two things. Three things. [excited] Big news.")
    say.aura_pcm(s, expressive=True)
    assert len(calls) == 4
    calls.clear()
    say.aura_pcm(s, expressive=True, flow=True)
    assert calls == ["One thing. Two things. Three things.", "Big news!"]


def test_aura_flow_off_in_live_mode(monkeypatch):
    from paralang import aura
    calls = []
    monkeypatch.setattr(aura, "synth", _fake_aura(calls))
    say.aura_pcm(express("[calm] One. Two. Three."), flow=True, live=True)
    assert len(calls) == 3


def test_expressive_default_off(monkeypatch):
    monkeypatch.delenv("PARALANG_EXPRESSIVE", raising=False)
    assert say._expressive(None) is False
    monkeypatch.setenv("PARALANG_EXPRESSIVE", "1")
    assert say._expressive(None) is True
    assert say._expressive(False) is False


def test_plain_aura_unchanged(monkeypatch):
    """expressive off = byte-for-byte the old concat (shaping off too)."""
    from paralang import aura
    calls = []
    monkeypatch.setattr(aura, "synth", _fake_aura(calls))
    monkeypatch.delenv("PARALANG_EXPRESSIVE", raising=False)
    s = express("[excited] Confirmed. See you.")
    pcm = say.aura_pcm(s)
    assert calls == ["Confirmed.", "See you."]
    clip = _tone_pcm(0.2, lead=0.1, tail=0.1)
    gap = b"\x00\x00" * (SR * s.lines[1].prosody.pause_before_ms // 1000)
    assert pcm == clip + gap + clip


def test_directions_acted_by_orpheus_never_read_aloud_elsewhere():
    from paralang.tts import to_aura, to_edge, to_elevenlabs, to_gemini, to_openai, to_orpheus, to_ssml
    s = express("[warm] Okay. [sigh] That was a long day. [dark chuckle] Anyway.")
    assert "[sigh]" in to_orpheus(s)[0] and "[dark chuckle]" in to_orpheus(s)[0]
    assert "[sigh]" in to_elevenlabs(s)["text"]
    for text in ([t for t, _ in to_aura(s)] + [t for t, _, _ in to_edge(s)]
                 + [to_ssml(s), to_openai(s)["input"], to_gemini(s)]):
        assert "sigh" not in text and "chuckle" not in text
