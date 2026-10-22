import json
from datetime import datetime, timezone

from paralang import report


def call(ts="2026-10-20T10:00:00+00:00"):
    return {
        "ts": ts, "result": {"tier": "warm", "score": 55},
        "turn_timing": {"reply_gaps": [0.8, 1.1, 0.9], "cut_offs": 1},
        "transcript": [
            {"role": "assistant", "content": "Namaste ji, main Riya, Jio se."},
            {"role": "user", "content": "haan ji bolo"},
            {"role": "assistant", "content": "Naya plan sirf Rs 299 ka hai, 2GB roz."},
            {"role": "user", "content": "achha... sochta hoon"},
            {"role": "assistant", "content": "Main WhatsApp pe details bhej deti hoon?"},
            {"role": "user", "content": "haan bhej do haha"},
        ],
        "paralang": [
            {"turn": 1, "text": "haan ji bolo", "cues": []},
            {"turn": 2, "text": "achha... sochta hoon", "cues": [{"name": "hesitant", "strength": 0.8},
                                                                  {"name": "flat", "strength": 0.6}]},
            {"turn": 3, "text": "haan bhej do haha", "cues": [{"name": "laughed", "strength": 0.7}]},
        ],
    }


def test_moments_pair_the_change_with_the_agent_line():
    ms = report.moments(call())
    assert any("turn 2" in m and "hesitant" in m and "price" in m for m in ms)
    assert any("dropped" in m and "price" in m for m in ms)
    assert any("turn 3" in m and "laughed" in m for m in ms)


def test_call_report_has_svg_table_and_timing():
    md = report.call_report(call())
    assert "<svg" in md and "| 2 |" in md and "cut-offs 1" in md and "warm 55" in md


def test_weekly_rollup_and_cli(tmp_path):
    log = tmp_path / "call_log.jsonl"
    log.write_text("\n".join(json.dumps(c) for c in (call(), call(), {"no": "paralang"})) + "\nnot json\n",
                   encoding="utf-8")
    calls = report.load_calls(log)
    assert len(calls) == 2
    md = report.weekly(calls, now=datetime(2026, 10, 22, tzinfo=timezone.utc))
    assert "2 calls" in md and "after price: 2 calls" in md and "hesitant: 100%" in md
    out = tmp_path / "weekly.md"
    report.main([str(log), "--week", "--days", "3650", "--out", str(out)])
    assert out.read_text(encoding="utf-8").startswith("# Weekly voice report")
