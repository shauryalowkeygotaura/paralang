import json
from types import SimpleNamespace as NS
from pathlib import Path

import pytest

from paralang.playbook import Playbook, Rule

JIO = Path(__file__).resolve().parents[2] / "jio-outbound" / "campaign" / "playbook.json"


def turn(n, **cues):
    return NS(turn=n, cues=[NS(name=k, strength=v) for k, v in cues.items()])


def test_price_hesitation_needs_the_price_context():
    pb = Playbook([Rule.from_dict({"name": "p", "when": {"cue": ["hesitant"], "after_topic": "price"},
                                   "coach": "offer cheaper"})])
    assert pb.advise(turn(2, hesitant=0.8), "Kya aap naya plan lenge?") is None      # no price said
    assert pb.advise(turn(3, hesitant=0.8), "Ye plan sirf Rs 299 ka hai") == "offer cheaper"
    assert pb.advise(turn(4, hesitant=0.9), "Rs 299 hai") is None                       # once per call


def test_strength_turn_min_and_order():
    pb = Playbook([Rule.from_dict({"name": "a", "when": {"cue": "flat", "turn_min": 2}, "coach": "A"}),
                   Rule.from_dict({"name": "b", "when": {"cue": "flat"}, "coach": "B"})])
    assert pb.advise(turn(1, flat=0.9)) == "B"           # a needs turn >= 2
    assert pb.advise(turn(2, flat=0.3)) is None          # too weak for both
    assert pb.advise(turn(3, flat=0.9)) == "A"


def test_bad_rules_rejected():
    with pytest.raises(ValueError):
        Rule.from_dict({"name": "x"})
    with pytest.raises(ValueError):
        Rule.from_dict({"name": "x", "coach": "c", "when": {"after_topic": "weather"}})


@pytest.mark.skipif(not JIO.exists(), reason="jio-outbound not checked out")
def test_jio_playbook_loads_and_fires():
    pb = Playbook.load(JIO)
    assert len(pb.rules) >= 6
    assert "cheaper" in pb.advise(turn(3, long_think=0.9), "Naya plan Rs 349 ka hai")
