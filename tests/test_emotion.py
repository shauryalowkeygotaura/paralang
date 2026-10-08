import xml.dom.minidom

import pytest

from paralang import EMOTIONS, Cue, express
from paralang.emotion import parse
from paralang.tts import to_edge, to_elevenlabs, to_gemini, to_openai, to_ssml


def test_markup_switches_emotion_and_intensity():
    segs = parse("[warm] Namaste ji. Kaise hain? [excited:0.5] Bada offer hai! [nope] stays.")
    assert [(e, k) for e, k, _ in segs] == [("warm", 1.0), ("warm", 1.0), ("excited", 0.5), ("excited", 0.5)]
    assert segs[-1][2] == "[nope] stays."          # unknown tags untouched (may be for the TTS)


def test_emotions_move_prosody_the_right_way():
    def p(e):
        return express("Ye ek line hai.", e, wobble=False).lines[0].prosody
    assert p("excited").rate_pct > p("neutral").rate_pct > p("sad").rate_pct
    assert p("whisper").volume_pct < p("neutral").volume_pct
    assert p("excited").pitch_hz > p("calm").pitch_hz


def test_intensity_and_cues_stack():
    half = express("Line.", "excited", 0.5, wobble=False).lines[0].prosody
    full = express("Line.", "excited", 1.0, wobble=False).lines[0].prosody
    assert 0 < half.rate_pct < full.rate_pct
    calmer = express("Line.", "warm", wobble=False, cues=[Cue("hesitant", 1.0)]).lines[0].prosody
    assert calmer.rate_pct < express("Line.", "warm", wobble=False).lines[0].prosody.rate_pct


def test_dramatic_pauses_longer_than_urgent():
    t = "Ek. Do. Teen."
    d = [l.prosody.pause_before_ms for l in express(t, "dramatic", wobble=False).lines]
    u = [l.prosody.pause_before_ms for l in express(t, "urgent", wobble=False).lines]
    assert d[0] == u[0] == 0 and d[1] > u[1] * 2


def test_bad_input_rejected():
    with pytest.raises(ValueError):
        express("hi", "furious")
    with pytest.raises(ValueError):
        express("[warm]")


def test_renderers():
    s = express("[warm] Namaste ji. [excited] Offer <50% off> hai!")
    edge = to_edge(s)
    assert len(edge) == 2 and edge[1][1]["rate"].endswith("%")
    for azure in (False, True):
        ssml = to_ssml(s, "en-IN-NeerjaNeural", azure=azure)
        xml.dom.minidom.parseString(ssml)            # well-formed, '<' and '%' escaped
        assert ("express-as" in ssml) == azure
    el = to_elevenlabs(s)
    assert el["text"].startswith("[warmly] Namaste ji. [excited]") and el["voice_settings"]["stability"] < 0.5
    oa = to_openai(s)
    assert "warm" in oa["instructions"] and "excited" in oa["instructions"] and oa["input"] == s.text
    assert to_gemini(express("Hi there.", "sad")).startswith("Speak sad and subdued")


def test_every_emotion_is_complete():
    for name, e in EMOTIONS.items():
        assert e.describe, name
        express("Test line.", name)                   # renders without error
