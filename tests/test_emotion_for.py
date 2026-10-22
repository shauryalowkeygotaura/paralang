from paralang.cues import Cue
from paralang.delivery import emotion_for


def test_agitated_beats_rushed_even_on_a_tie():
    cues = [Cue("rushed", 1.0, {}), Cue("agitated", 1.0, {})]
    assert emotion_for(cues) == "calm"


def test_strongest_cue_else_base():
    assert emotion_for([Cue("hesitant", 0.7, {}), Cue("rushed", 0.5, {})]) == "empathetic"
    assert emotion_for([Cue("hesitant", 0.2, {})]) == "warm"
