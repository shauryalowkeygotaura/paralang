"""Beats (`|`), emphasis (`*word*`), and the Orpheus / Aura targets."""
from paralang.emotion import clean, express
from paralang.tts import to_aura, to_edge, to_orpheus, to_ssml

S = "[serious] Four rocks are coming. [dramatic] And none of them | are *theoretical*. [serious] One has a name. *Apophis*."


def test_clean_strips_marks():
    assert clean("And none of them | are *theoretical*.") == "And none of them are theoretical."


def test_line_text_is_clean_and_beats_keep_order():
    s = express(S)
    line = s.lines[1]
    assert line.text == "And none of them are theoretical."
    assert line.beats() == [("And none of them", False), ("are", False), ("theoretical.", True)]


def test_punctuation_never_becomes_its_own_beat():
    s = express("[dramatic] It is *real*.")
    assert s.lines[0].beats() == [("It is", False), ("real.", True)]


def test_text_engines_get_ellipsis_beats():
    s = express(S)
    assert to_edge(s)[1][0] == "And none of them... are... theoretical."
    assert to_aura(s)[1][0] == "And none of them... are... theoretical."


def test_ssml_gets_real_break_and_emphasis():
    x = to_ssml(express(S))
    assert '<break time="420ms"/>are' in x
    assert '<emphasis level="moderate">theoretical.</emphasis>' in x


def test_orpheus_one_chunk_with_directions_at_changes_only():
    chunks = to_orpheus(express(S))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.startswith("[serious] Four rocks are coming.")
    assert "[dramatic] And none of them... are... theoretical." in c
    assert c.count("[serious]") == 2


def test_orpheus_chunks_respect_limit_and_restate_direction():
    long = "[dramatic] " + " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = to_orpheus(express(long), max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert all(c.startswith("[dramatic]") for c in chunks)


def test_aura_gaps_come_from_planner_and_first_is_zero():
    gaps = [p for _, p in to_aura(express(S))]
    assert gaps[0] == 0 and all(g > 0 for g in gaps[1:])


def test_neutral_has_no_orpheus_tag():
    assert to_orpheus(express("Just a plain line."))[0] == "Just a plain line."
