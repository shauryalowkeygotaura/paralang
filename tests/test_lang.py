"""On-the-fly English <-> Hindi voice switching."""
import pytest

from paralang import express
from paralang import say
from paralang.lang import ReplyRouter, detect_lang


@pytest.mark.parametrize("text,lang", [
    ("Your appointment is confirmed for tomorrow at ten.", "en"),
    ("This is Maya, works for me.", "en"),                # voicezero's known false-positive guard
    ("Aapka plan kal khatam ho raha hai.", "hi"),
    ("Koi baat nahi ji, aap aaram se sochiye.", "hi"),
    ("आपका प्लान कल खत्म होगा।", "hi"),
])
def test_detect_matches_voicezero(text, lang):
    assert detect_lang(text) == lang


def test_matches_voicezero_source_exactly():
    # both systems must route the same reply the same way
    import importlib.util
    from pathlib import Path
    vz = Path(__file__).resolve().parents[2] / "voicezero" / "tts_providers.py"
    if not vz.exists():
        pytest.skip("voicezero not checked out")
    spec = importlib.util.spec_from_file_location("vz_tts", vz)
    src = vz.read_text(encoding="utf-8")
    from paralang import lang
    import re
    words = re.search(r'_HINDI_ROMAN = frozenset\(\((.*?)\)\.split\(\)\)', src, re.S).group(1)
    vz_set = set(" ".join(re.findall(r'"([^"]*)"', words)).split())
    assert vz_set == set(lang._HINDI_ROMAN)


def test_router_sticky_then_released():
    r = ReplyRouter()
    assert r.lang("Achha ji.", "r1", caller_text="Mera plan kab khatam hoga?") == "hi"   # short: mirror caller
    assert r.lang("Your plan ends tomorrow.", "r1") == "hi"                           # sticky (short)
    assert r.lang("I will send the full details on WhatsApp right now.", "r1") == "en"  # clearly English: release
    assert r.lang("Okay.", "r2", caller_text="Yes please send it") == "en"            # new reply, English caller


def test_aura_pcm_switches_hindi_to_swara(monkeypatch):
    from paralang import aura
    used = []
    monkeypatch.setattr(aura, "synth", lambda t, v, live=False: used.append(("aura", t)) or b"AU")

    async def edge(t, v, kw):
        used.append((v, t))
        return b"ED"
    monkeypatch.setattr(say, "_synth", edge)
    s = express("Hello, this is Riya calling from the clinic today. Aapka appointment kal subah hai. "
                "Koi baat nahi ji. Please call us back if anything changes at all.", wobble=False)
    say.aura_pcm(s)
    assert [u[0] for u in used] == ["aura", "hi-IN-SwaraNeural", "hi-IN-SwaraNeural", "aura"]
