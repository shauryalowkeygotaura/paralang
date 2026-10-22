"""
tune.py -- improve paralang over time from real calls (v0.3 #13b).

    python -m paralang.tune call_log.jsonl [more.jsonl ...] --out Projects/paralang/tuning-2026-10-26.md

Reads the call dumps the agents already write and PROPOSES changes as a
reviewed Markdown file. It never edits thresholds, playbooks or prompts itself:
a wrong automatic change on live calls costs real leads, so a human approves
every diff. What it looks at:

  cues       share of calls each cue fires on, and outcome with vs without it
             (a cue on > 60% of calls is probably too sensitive; one that never
             moves outcomes is noise)
  playbook   how often each rule fired, and the outcome of calls where it did
  timing     agent reply-gap median and cut-off rate (smart-turn / VAD tuning)
  self-check how often the agent was too fast / monotone / slow to answer
  pitch      the fast numpy F0 vs openSMILE F0 stored beside it (tracker health)

Below MIN_CALLS calls it still reports, but every proposal is marked
"insufficient data".
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from .report import load_calls

MIN_CALLS = 20
TIER_RANK = {"dnd": 0, "dead": 0, "cold": 1, "warm": 2, "hot": 3}
TOO_COMMON = 0.6
CUT_OFF_RATE_HIGH = 0.10
REPLY_GAP_HIGH_S = 2.0
SELF_FLAG_HIGH = 0.30
F0_DRIFT_ST = 1.0


def _rank(call: dict) -> int | None:
    return TIER_RANK.get((call.get("result") or {}).get("tier"))


def _good(call: dict) -> bool:
    r = _rank(call)
    return r is not None and r >= 2


def analyse(calls: list[dict]) -> dict:
    n = len(calls)
    cue_calls: dict[str, list[dict]] = defaultdict(list)
    for c in calls:
        names = {cu["name"] for t in c.get("paralang", []) for cu in t.get("cues", [])
                 if cu.get("strength", 0) >= 0.4}
        for name in names:
            cue_calls[name].append(c)
    base_good = sum(_good(c) for c in calls) / n if n else 0.0

    cues = {}
    for name, with_cue in cue_calls.items():
        good_with = sum(_good(c) for c in with_cue) / len(with_cue)
        without = [c for c in calls if c not in with_cue]
        good_without = sum(_good(c) for c in without) / len(without) if without else None
        cues[name] = {"share": len(with_cue) / n, "good_with": good_with, "good_without": good_without}

    rules = defaultdict(list)
    for c in calls:
        for note in c.get("coached", []) or []:
            rules[note.get("coach", "")[:60]].append(c)
    playbook = {k: {"fired": len(v), "good": sum(_good(c) for c in v) / len(v)} for k, v in rules.items()}

    gaps = [g for c in calls for g in (c.get("turn_timing") or {}).get("reply_gaps", [])]
    turns = sum(len(c.get("paralang", [])) for c in calls)
    cut_offs = sum((c.get("turn_timing") or {}).get("cut_offs", 0) for c in calls)

    flags = Counter()
    replies = 0
    for c in calls:
        for r in c.get("selfcheck", []) or []:
            replies += 1
            flags.update(r.get("flags", []))

    drift = []
    for c in calls:
        for t in c.get("paralang", []):
            f = t.get("features", {})
            smile = (f.get("extras") or {}).get("smile_f0_median_st")
            if smile is not None and f.get("f0_median_st") is not None:
                drift.append(abs(f["f0_median_st"] - smile))

    return {"n": n, "base_good": base_good, "cues": cues, "playbook": playbook,
            "gap_median": statistics.median(gaps) if gaps else None,
            "cut_off_rate": cut_offs / turns if turns else None,
            "self_flags": {k: v / replies for k, v in flags.items()} if replies else {},
            "replies": replies,
            "f0_drift_st": statistics.median(drift) if drift else None}


def proposals(a: dict) -> list[str]:
    out = []
    for name, s in sorted(a["cues"].items(), key=lambda kv: -kv[1]["share"]):
        if s["share"] > TOO_COMMON:
            out.append(f"`{name}` fires on {s['share']:.0%} of calls: probably too sensitive. Raise its "
                       f"threshold in `paralang/cues.py` (or its _ramp start) and re-run this report.")
        if s["good_without"] is not None and abs(s["good_with"] - s["good_without"]) < 0.05 and s["share"] > 0.15:
            out.append(f"`{name}` does not move outcomes (warm+ {s['good_with']:.0%} with vs "
                       f"{s['good_without']:.0%} without). Consider dropping its scoring weight.")
    for rule, s in a["playbook"].items():
        if s["fired"] >= 5 and s["good"] < a["base_good"] - 0.10:
            out.append(f"Playbook note \"{rule}...\" fired {s['fired']}x with warm+ {s['good']:.0%} "
                       f"(baseline {a['base_good']:.0%}): rewrite or drop it in the campaign playbook.json.")
    if a["cut_off_rate"] is not None and a["cut_off_rate"] > CUT_OFF_RATE_HIGH:
        out.append(f"Agent cut callers off on {a['cut_off_rate']:.0%} of turns: raise the VAD stop_secs "
                   f"(0.2 -> 0.4) or smart-turn's stop_secs so it waits longer before replying.")
    if a["gap_median"] is not None and a["gap_median"] > REPLY_GAP_HIGH_S:
        out.append(f"Median reply gap {a['gap_median']:.2f}s: callers wait too long. Check TTS first-audio "
                   f"latency (Andromeda cap PARALANG_LIVE_FIRST_AUDIO_S) and LLM latency.")
    sf = a["self_flags"]
    if sf.get("too_fast", 0) > SELF_FLAG_HIGH:
        out.append(f"Agent spoke much faster than the caller in {sf['too_fast']:.0%} of replies: lower the "
                   f"base speaking rate (TTS_RATE / DeliveryPlanner base_rate_pct).")
    if sf.get("monotone", 0) > SELF_FLAG_HIGH:
        out.append(f"Agent was monotone in {sf['monotone']:.0%} of replies: use an engine that acts emotion "
                   f"(PARALANG_ENGINE=orpheus) for English, or check the delivery wobble.")
    if sf.get("dead_air", 0) > SELF_FLAG_HIGH:
        out.append(f"Dead air before {sf['dead_air']:.0%} of replies: see the reply-gap note.")
    if a["f0_drift_st"] is not None and a["f0_drift_st"] > F0_DRIFT_ST:
        out.append(f"Fast pitch tracker disagrees with openSMILE by {a['f0_drift_st']:.1f} semitones "
                   f"(median): check octave errors on real phone audio before trusting pitch cues.")
    return out


def render(a: dict, sources: list[str]) -> str:
    enough = a["n"] >= MIN_CALLS
    lines = [f"# paralang tuning proposal ({date.today().isoformat()})", "",
             f"Sources: {', '.join(sources)} | calls: {a['n']} | warm+ rate {a['base_good']:.0%}", "",
             "**Nothing here has been applied.** Review, then edit the files named in each line.", ""]
    if not enough:
        lines += [f"> Insufficient data: {a['n']} calls (need {MIN_CALLS}). Proposals below are "
                  f"directional only.", ""]
    props = proposals(a)
    lines += ["## Proposals", *([f"- {p}" for p in props] or ["- None: everything is inside its thresholds."]), ""]
    lines += ["## Cues", "| cue | share of calls | warm+ with | warm+ without |", "|---|---|---|---|"]
    for name, s in sorted(a["cues"].items(), key=lambda kv: -kv[1]["share"]):
        wo = f"{s['good_without']:.0%}" if s["good_without"] is not None else "-"
        lines.append(f"| {name} | {s['share']:.0%} | {s['good_with']:.0%} | {wo} |")
    lines += ["", "## Timing and self-check",
              f"- reply gap median: {a['gap_median']:.2f}s" if a["gap_median"] is not None else "- reply gap: no data",
              f"- cut-off rate: {a['cut_off_rate']:.0%} of turns" if a["cut_off_rate"] is not None else "- cut-offs: no data",
              f"- agent self-check over {a['replies']} replies: "
              + (", ".join(f"{k} {v:.0%}" for k, v in a["self_flags"].items()) or "no flags"),
              f"- pitch tracker drift vs openSMILE: {a['f0_drift_st']:.2f} st" if a["f0_drift_st"] is not None
              else "- pitch tracker drift: no openSMILE pairs yet",
              "", "#research #paralang #tuning"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="paralang.tune")
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    calls = []
    for p in a.logs:
        if p.exists():
            try:
                calls += load_calls(p)
            except SystemExit as e:
                print(f"skip {p}: {e}", file=sys.stderr)
    if not calls:
        md = (f"# paralang tuning proposal ({date.today().isoformat()})\n\nNo calls with paralang data yet "
              f"in {', '.join(map(str, a.logs))}. Run real calls first (the live mic test is still open).\n")
    else:
        md = render(analyse(calls), [str(p) for p in a.logs])
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md + "\n", encoding="utf-8")
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
