"""
report.py -- the emotion arc of a call, and a weekly roll-up (v0.3 #10).

    python -m paralang.report call_log.jsonl                 # last call
    python -m paralang.report call_log.jsonl --call 3        # 4th call in the file (0-based)
    python -m paralang.report call_log.jsonl --week          # roll-up, last 7 days
    python -m paralang.report call_log.jsonl --week --out Projects/x/weekly.md

Input: the JSONL call dump the agents already write. Each line is a call with a
"paralang" list (Analyzer.export()) and, when available, "transcript",
"result", "turn_timing", "events", "ts". Nothing new needs capturing.

Output: Markdown with an inline SVG engagement line (renders in Obsidian), a
per-turn table, and "moments": the turn where something changed, next to what
the AGENT had just said ("went flat right after the price"). That pairing is
the point: it tells you which line to rewrite. This is what the retainer review
shows a clinic every week.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# How each cue moves engagement (per unit strength). Situation cues are 0.
VALENCE = {"laughed": 1.0, "rushed": -0.5, "hesitant": -1.0, "flat": -1.0, "agitated": -1.5,
           "long_think": -0.5, "uncertain_rise": -0.3, "sighed": -1.0, "barged_in": -0.3,
           "noisy": 0.0, "machine": 0.0}
TOPICS = {
    "price": re.compile(r"(\brs\.?\s?\d|₹|\brupa?y|\bprice\b|\bkitne\b|\bcost\b|\bfees?\b|\d+\s?(rs|rupees))", re.I),
    "offer": re.compile(r"\b(offer|plan|upgrade|recharge|discount|scheme)\b", re.I),
    "booking": re.compile(r"\b(appointment|book|slot|tomorrow|kal|time)\b", re.I),
    "whatsapp": re.compile(r"\bwhats\s?app\b", re.I),
}


def load_calls(path: Path) -> list[dict]:
    calls = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise SystemExit(f"cannot read {path}: {e}")
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            print(f"warn: line {n} is not JSON; skipped", file=sys.stderr)
            continue
        if isinstance(rec, dict) and rec.get("paralang"):
            calls.append(rec)
    return calls


def engagement(turn: dict) -> float:
    s = sum(VALENCE.get(c["name"], 0.0) * c.get("strength", 0.0) for c in turn.get("cues", []))
    return max(0.0, min(100.0, 60.0 + 25.0 * s))


def topics(text: str) -> list[str]:
    return [k for k, rx in TOPICS.items() if rx.search(text or "")]


def agent_lines_before(call: dict) -> list[str]:
    """For caller turn i (paralang order), the agent line spoken just before it."""
    out, last = [], ""
    for m in call.get("transcript", []) or []:
        if m.get("role") == "assistant":
            last = m.get("content", "")
        elif m.get("role") == "user":
            out.append(last)
    return out


def moments(call: dict) -> list[str]:
    turns = call["paralang"]
    before = agent_lines_before(call)
    seen: set[str] = set()
    out = []
    prev = None
    for i, t in enumerate(turns):
        e = engagement(t)
        names = {c["name"] for c in t.get("cues", []) if c.get("strength", 0) >= 0.4}
        new = sorted(names - seen)
        said = before[i] if i < len(before) else ""
        about = topics(said)
        ctx = (f" right after the agent mentioned {' + '.join(about)}" if about else "")
        if new:
            out.append(f"turn {t['turn']}: first **{', '.join(new)}**{ctx}"
                       + (f' (agent: "{said[:80]}")' if said else ""))
        if prev is not None and e - prev <= -20:
            out.append(f"turn {t['turn']}: engagement dropped {prev:.0f} -> {e:.0f}{ctx}")
        elif prev is not None and e - prev >= 20:
            out.append(f"turn {t['turn']}: engagement recovered {prev:.0f} -> {e:.0f}{ctx}")
        seen |= names
        prev = e
    return out


def sparkline(values: list[float], w: int = 360, h: int = 60) -> str:
    if not values:
        return ""
    n = max(1, len(values) - 1)
    pts = " ".join(f"{i * w / n:.1f},{h - v / 100 * h:.1f}" for i, v in enumerate(values))
    dots = "".join(f'<circle cx="{i * w / n:.1f}" cy="{h - v / 100 * h:.1f}" r="3" fill="#7a1f2b"/>'
                   for i, v in enumerate(values))
    return (f'<svg width="{w}" height="{h}" viewBox="-4 -4 {w + 8} {h + 8}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="0" y1="{h - 0.6 * h:.1f}" x2="{w}" y2="{h - 0.6 * h:.1f}" stroke="#ccc" stroke-dasharray="3 3"/>'
            f'<polyline fill="none" stroke="#7a1f2b" stroke-width="2" points="{pts}"/>{dots}</svg>')


def call_report(call: dict) -> str:
    turns = call["paralang"]
    values = [engagement(t) for t in turns]
    res = call.get("result", {}) or {}
    timing = call.get("turn_timing", {}) or {}
    gaps = timing.get("reply_gaps") or []
    lines = [f"## Call {call.get('ts', '')[:16]}  {res.get('tier', '')} {res.get('score', '')}".rstrip(),
             "", sparkline(values), "",
             f"Engagement start {values[0]:.0f} -> end {values[-1]:.0f} over {len(turns)} caller turns"
             + (f"; agent reply gap median {sorted(gaps)[len(gaps) // 2]:.2f}s" if gaps else "")
             + (f"; cut-offs {timing.get('cut_offs')}" if timing.get("cut_offs") else ""), ""]
    ms = moments(call)
    if ms:
        lines += ["**Moments**", *[f"- {m}" for m in ms], ""]
    lines += ["| turn | engagement | cues | caller said |", "|---|---|---|---|"]
    for t, v in zip(turns, values):
        cues = ", ".join(f"{c['name']} {c['strength']:.1f}" for c in t.get("cues", [])) or "-"
        text = (t.get("text") or "").replace("|", "/")[:60]
        lines.append(f"| {t['turn']} | {v:.0f} | {cues} | {text} |")
    return "\n".join(lines)


def weekly(calls: list[dict], days: int = 7, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    recent = []
    for c in calls:
        try:
            ts = datetime.fromisoformat(c.get("ts", ""))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if now - ts <= timedelta(days=days):
            recent.append(c)
    if not recent:
        return f"# Weekly voice report\n\nNo calls in the last {days} days."
    cue_calls = Counter()
    drops = Counter()
    machine = 0
    for c in recent:
        names = {cu["name"] for t in c["paralang"] for cu in t.get("cues", []) if cu.get("strength", 0) >= 0.4}
        cue_calls.update(names)
        machine += "machine" in names
        for m in moments(c):
            for topic in TOPICS:
                if "dropped" in m and topic in m:
                    drops[topic] += 1
    ends = [engagement(c["paralang"][-1]) for c in recent]
    tiers = Counter((c.get("result") or {}).get("tier", "?") for c in recent)
    out = [f"# Weekly voice report ({len(recent)} calls, last {days} days)", "",
           f"- Answered by a machine: {machine} ({machine / len(recent):.0%})",
           f"- Ending engagement: median {sorted(ends)[len(ends) // 2]:.0f}/100",
           f"- Outcome tiers: " + ", ".join(f"{k} {v}" for k, v in tiers.most_common()), "",
           "**How callers sounded (share of calls)**",
           *[f"- {n}: {k / len(recent):.0%}" for n, k in cue_calls.most_common()], ""]
    if drops:
        out += ["**Where engagement dropped (agent topic just before)**",
                *[f"- after {t}: {k} calls" for t, k in drops.most_common()],
                "", "Rewrite the agent's lines on the top topic first.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="paralang.report")
    ap.add_argument("log", type=Path, help="call_log.jsonl")
    ap.add_argument("--call", type=int, help="0-based call index (default: last)")
    ap.add_argument("--week", action="store_true", help="roll-up of the last 7 days")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    calls = load_calls(a.log)
    if not calls:
        raise SystemExit(f"{a.log}: no calls with a paralang block")
    if a.week:
        md = weekly(calls, a.days)
    else:
        idx = a.call if a.call is not None else len(calls) - 1
        if not -len(calls) <= idx < len(calls):
            raise SystemExit(f"--call {idx} out of range: {len(calls)} calls")
        md = call_report(calls[idx])
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md + "\n", encoding="utf-8")
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
