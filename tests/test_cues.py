"""hesitant / flat / agitated: the two-signal rule, on hand-built features."""
from paralang.cues import derive_cues
from paralang.features import TurnFeatures


def feats(**kw):
    base = dict(duration_s=3.0, voiced_s=2.5, words=6, words_per_sec=2.4, pause_count=0,
                pause_total_s=0.0, longest_pause_s=0.0, pause_ratio=0.0, fillers=0)
    base.update(kw)
    return TurnFeatures(**base)


def names(f, z):
    return {c.name: round(c.strength, 2) for c in derive_cues(f, z)}


def test_calm_normal_turn_has_no_judgement_cues():
    n = names(feats(), {"words_per_sec": 0.1, "loudness": 0.2, "f0_range_st": -0.3, "f0_median_st": 0.1})
    assert not ({"hesitant", "flat", "agitated"} & n.keys())


def test_hesitant_needs_two_signals():
    # one 1.0 s pause alone (a Hinglish word search) is not hesitation
    assert "hesitant" not in names(feats(longest_pause_s=1.0, pause_count=1), {})
    # pause + filler is
    n = names(feats(longest_pause_s=1.2, pause_count=1, fillers=1), {})
    assert n.get("hesitant", 0) >= 0.4
    # filler + clearly slower than usual, no pause
    assert "hesitant" in names(feats(fillers=2), {"words_per_sec": -2.0})


def test_one_very_long_pause_is_weak_hesitation_in_a_real_sentence():
    n = names(feats(longest_pause_s=2.0, pause_count=1), {})
    assert 0.4 <= n["hesitant"] <= 0.6
    assert "hesitant" not in names(feats(longest_pause_s=2.0, words=2), {})   # "haan... ji"


def test_flat_needs_baseline_and_a_sentence():
    z = {"f0_range_st": -2.0, "loudness": -1.8}
    assert "flat" not in names(feats(), {})                 # no baseline yet: unknown, not flat
    assert "flat" not in names(feats(words=1), z)           # "haan" is always flat
    assert names(feats(), z)["flat"] >= 0.7                 # narrow + quiet
    lone = names(feats(), {"f0_range_st": -2.0, "loudness": 0.0})
    assert 0.4 <= lone["flat"] <= 0.5                        # pitch alone: weak


def test_agitated_requires_loudness():
    # excited lead: higher + faster but NOT louder -> not agitated
    assert "agitated" not in names(feats(), {"f0_median_st": 2.5, "words_per_sec": 2.5, "loudness": 0.3})
    # louder alone -> not agitated (could be a bad line / speakerphone)
    assert "agitated" not in names(feats(), {"loudness": 2.5})
    # louder + higher pitch -> agitated
    assert names(feats(), {"loudness": 2.5, "f0_median_st": 2.5})["agitated"] >= 0.6


def test_all_three_are_tagged_and_scored():
    from paralang import Analyzer
    from paralang.analyzer import PHRASE
    assert {"hesitant", "flat", "agitated"} <= PHRASE.keys()
    # the jio scorer already weighs these names
    import importlib.util
    from pathlib import Path
    sc = Path(__file__).resolve().parents[2] / "jio-outbound" / "campaign" / "intent_scoring.py"
    if sc.exists():
        src = sc.read_text(encoding="utf-8")
        assert '"hesitant"' in src and '"tone: flat"' in src and '"tone: agitated"' in src
