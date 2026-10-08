"""Auto-markup: the word guard and the tidy pass (no network)."""
from paralang import markup


def test_words_ignore_marks_but_catch_changes():
    a = "There are four rocks. None are theoretical."
    assert markup._words("[serious] There are four rocks. None are | *theoretical*.") == markup._words(a)
    assert markup._words("[serious] There are five rocks. None are theoretical.") != markup._words(a)


def test_tidy_drops_useless_beats_and_newlines():
    out = markup.tidy("[serious] Line one.|\n[dramatic] Not | theoretical.| Next |[calm] x.")
    assert out == "[serious] Line one. [dramatic] Not | theoretical. Next [calm] x."


def test_tidy_restores_swallowed_space_but_not_decimals():
    assert markup.tidy("That sounds small.It is not. A 2.7 percent chance.") == \
        "That sounds small. It is not. A 2.7 percent chance."


def test_annotate_keeps_paragraph_unmarked_when_words_change(monkeypatch):
    monkeypatch.setattr(markup, "_call", lambda para, key: "[serious] Totally different words.")
    assert markup.annotate("Four rocks are coming.", key="x") == "Four rocks are coming."


def test_annotate_uses_marked_text_when_words_match(monkeypatch):
    monkeypatch.setattr(markup, "_call", lambda para, key: "[dramatic] Four rocks are | *coming*.")
    assert markup.annotate("Four rocks are coming.", key="x") == "[dramatic] Four rocks are | *coming*."
