import json

import numpy as np

from paralang import Analyzer, tune
from paralang.selfcheck import SelfCheck

SR = 16000


def call(tier, cues, coached=None, cut_offs=0, gaps=(0.8,), selfflags=()):
    return {"ts": "2026-10-20T10:00:00+00:00", "result": {"tier": tier},
            "paralang": [{"turn": 1, "cues": [{"name": c, "strength": 0.8} for c in cues],
                          "features": {"f0_median_st": 30.0, "extras": {"smile_f0_median_st": 30.2}}}],
            "coached": coached or [], "turn_timing": {"reply_gaps": list(gaps), "cut_offs": cut_offs},
            "selfcheck": [{"flags": list(selfflags)}]}


def test_proposals_flag_oversensitive_cue_cutoffs_and_fast_agent():
    calls = ([call("cold", ["hesitant"], cut_offs=1, selfflags=["too_fast"]) for _ in range(15)]
             + [call("warm", []) for _ in range(5)])
    a = tune.analyse(calls)
    props = " ".join(tune.proposals(a))
    assert "`hesitant` fires on 75%" in props
    assert "cut callers off" in props and "faster than the caller" in props
    assert a["f0_drift_st"] < 1.0 and "disagrees" not in props


def test_render_marks_insufficient_data_and_never_applies(tmp_path):
    md = tune.render(tune.analyse([call("warm", ["laughed"])]), ["x.jsonl"])
    assert "Insufficient data" in md and "Nothing here has been applied" in md


def test_cli_handles_no_paralang_data(tmp_path):
    log = tmp_path / "c.jsonl"
    log.write_text(json.dumps({"ts": "x"}) + "\n", encoding="utf-8")
    out = tmp_path / "t.md"
    tune.main([str(log), "--out", str(out)])
    assert "No calls with paralang data yet" in out.read_text(encoding="utf-8")


def test_selfcheck_flags_fast_and_monotone_agent():
    caller = Analyzer(acoustic=False)
    t = np.arange(SR * 2) / SR
    tone = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    caller.analyze_turn(tone, SR, "haan ji theek hai")               # ~2 words/s
    sc = SelfCheck(caller)
    flat4 = (0.3 * np.sin(2 * np.pi * 200 * np.arange(SR * 4) / SR)).astype(np.float32)
    rec = sc.observe_reply(flat4, SR, " ".join(["word"] * 24), reply_gap_s=3.1)  # 6 words/s, flat
    assert set(rec["flags"]) == {"too_fast", "monotone", "dead_air"}
