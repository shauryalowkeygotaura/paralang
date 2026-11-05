"""Fish-style free-form cues: resolution, parsing, per-engine rendering, sound
bank gate and splicing. No network."""
import numpy as np
import pytest

from paralang import say, sounds
from paralang.direction import resolve
from paralang.emotion import EMOTIONS, express
from paralang.tts import parts, to_aura, to_edge, to_fish, to_orpheus, to_ssml


@pytest.mark.parametrize("phrase,kind,emotion,intensity", [
    ("warm", "emotion", "warm", None),
    ("excited:0.5", "emotion", "excited", 0.5),
    ("slightly sad", "emotion", "sad", 0.5),
    ("extremely angry", "emotion", "angry", 1.8),
    ("nervous", "emotion", "nervous", None),
    ("happy", "emotion", "cheerful", None),
    ("warm and happy", "emotion", "warm", None),
    ("whispering", "tone", None, None),
    ("soft tone", "tone", None, None),
    ("in a hurry tone", "tone", None, None),
    ("laughing nervously", "sound", None, None),
    ("exasperated sigh", "sound", None, None),
    ("long-break", "break", None, None),
    ("building suspense", "unknown", None, None),
])
def test_resolve(phrase, kind, emotion, intensity):
    d = resolve(phrase)
    assert d.kind == kind and d.emotion == emotion and d.intensity == intensity


def test_every_mapped_emotion_exists():
    from paralang.direction import EMOTION_WORDS
    assert set(EMOTION_WORDS.values()) <= set(EMOTIONS)


def test_free_form_cues_drive_prosody():
    def line(t):
        return express(t, wobble=False).lines[0]
    sad, slightly = line("[sad] I miss you."), line("[slightly sad] I miss you.")
    assert sad.emotion == slightly.emotion == "sad" and slightly.intensity == 0.5
    assert sad.prosody.rate_pct < slightly.prosody.rate_pct < 0
    shout, whisper = line("[angry][shouting] Out!"), line("[sad][whispering] I miss you.")
    assert shout.cue == "angry][shouting" and shout.prosody.volume_pct > 10
    assert whisper.cue == "sad][whispering" and whisper.prosody.volume_pct <= -15


def test_emphasis_and_break_cues():
    l = express("This is [emphasis] really important. Wait [break] what?").lines
    assert "*really*" in l[0].marked and "|" in l[1].marked


def test_cue_only_tail_rides_on_previous_line():
    s = express("[warm] That is all. [laughing]")
    assert len(s.lines) == 1 and "[laughing]" in s.lines[0].marked


def test_acting_engines_get_cues_verbatim():
    s = express("[slightly sad] I miss you. [angry][shouting] Out! [laughing] Ha, fine.")
    (o,) = to_orpheus(s)
    assert o.startswith("[slightly sad] I miss you.") and "[angry][shouting] Out!" in o
    assert "[laughing] Ha, fine." in o
    (f,) = to_fish(s)
    assert f.startswith("[slightly sad] I miss you.") and "[laughing]" in f


def test_knob_engines_never_read_cues():
    s = express("[slightly sad] I miss you. [laughing] Ha, fine. [long-break] [building suspense] Go.")
    for text in [t for t, _ in to_aura(s)] + [t for t, _, _ in to_edge(s)]:
        assert "[" not in text and "laugh" not in text and "suspense" not in text
    assert '<break time="900ms"/>' in to_ssml(s)


def test_parts():
    assert parts("[laughing] Ha, fine.") == [("sound", "laugh"), ("text", "Ha, fine.")]
    assert parts("Wait. [long-break] Now [sighing].") == [
        ("text", "Wait."), ("pause", 900), ("text", "Now."), ("sound", "sigh")]
    assert parts("[building suspense] Go.") == [("text", "Go.")]


def test_onomatopoeia_gate():
    assert sounds.heard_words_ok("Ha ha ha.") and sounds.heard_words_ok("you") and sounds.heard_words_ok("")
    assert not sounds.heard_words_ok("Laughing.") and not sounds.heard_words_ok("I sigh")


def test_performer_matches_gender():
    assert sounds.performer_for("aura-2-andromeda-en") == "hannah"
    assert sounds.performer_for("en-IN-PrabhatNeural") == "troy"
    assert sounds.performer_for("troy") == "troy"


def test_sound_bank_cache_and_live_never_builds(tmp_path, monkeypatch):
    monkeypatch.setenv("PARALANG_SOUNDS_DIR", str(tmp_path))
    built = []
    monkeypatch.setattr(sounds, "build_one", lambda s, p, i, live=False: built.append((s, p, i)) or None)
    assert sounds.get("laugh", "aura-2-andromeda-en", live=True) is None and built == []
    assert sounds.get("laugh", "aura-2-andromeda-en") is None and len(built) == 3   # every variant tried
    path = tmp_path / "hannah" / "laugh_1.pcm"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x01\x00" * 10)
    assert sounds.get("laugh", "aura-2-andromeda-en", live=True) == b"\x01\x00" * 10
    assert sounds.get("nonsense") is None


def _tone(seconds):
    t = np.arange(int(24000 * seconds)) / 24000
    return (np.sin(2 * np.pi * 200 * t) * 0.3 * 32767).astype("<i2").tobytes()


def test_aura_splices_real_sound(monkeypatch):
    from paralang import aura
    calls, laugh = [], _tone(0.5)
    monkeypatch.setattr(aura, "synth", lambda text, voice, live=False: calls.append(text) or _tone(0.3))
    monkeypatch.setattr(sounds, "get", lambda name, voice=None, seed="", live=False: laugh if name == "laugh" else None)
    pcm = say.aura_pcm(express("[warm] Okay. [laughing] Ha, fine."), expressive=False)
    assert calls == ["Okay.", "Ha, fine."]
    assert laugh in pcm                                    # the clip is in the render, untouched


def test_missing_sound_is_skipped_not_spoken(monkeypatch):
    from paralang import aura
    calls = []
    monkeypatch.setattr(aura, "synth", lambda text, voice, live=False: calls.append(text) or _tone(0.2))
    monkeypatch.setattr(sounds, "get", lambda *a, **k: None)
    say.aura_pcm(express("[sighing] Long day."))
    assert calls == ["Long day."]
