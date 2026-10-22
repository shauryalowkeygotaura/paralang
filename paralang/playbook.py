"""
playbook.py -- cue-triggered coaching for the NEXT agent reply (v0.3 #11).

Rules are data (one JSON file per campaign), evaluated deterministically after
each caller turn. The first matching rule becomes a one-turn "[coach: ...]" note
in front of the caller's transcript, which the LLM follows and never reads
aloud (same convention as the [tone: ...] tag). Each rule fires at most once per
call unless "once": false, so the agent is steered, not nagged.

Rule schema:
    {"name": "price_hesitation",
     "when": {"cue": ["hesitant", "long_think"],   # any of; omit = no cue needed
              "min_strength": 0.5,                  # default 0.4
              "after_topic": "price",               # agent's last line was about it (report.TOPICS)
              "turn_min": 2},                       # optional
     "coach": "Offer the smaller pack once. Do not repeat the price.",
     "once": true}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .report import TOPICS


@dataclass
class Rule:
    name: str
    coach: str
    cues: list[str] = field(default_factory=list)
    min_strength: float = 0.4
    after_topic: str | None = None
    turn_min: int = 1
    once: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        if not isinstance(d, dict) or not d.get("name") or not d.get("coach"):
            raise ValueError(f"rule needs 'name' and 'coach': {d!r}")
        w = d.get("when", {}) or {}
        cue = w.get("cue", [])
        topic = w.get("after_topic")
        if topic is not None and topic not in TOPICS:
            raise ValueError(f"rule {d['name']}: unknown topic {topic!r} (have {', '.join(TOPICS)})")
        return cls(name=str(d["name"]), coach=str(d["coach"])[:300],
                   cues=[cue] if isinstance(cue, str) else list(cue),
                   min_strength=float(w.get("min_strength", 0.4)), after_topic=topic,
                   turn_min=int(w.get("turn_min", 1)), once=bool(d.get("once", True)))


class Playbook:
    def __init__(self, rules: list[Rule]):
        self.rules = rules
        self.fired: list[str] = []

    @classmethod
    def load(cls, path: str | Path) -> "Playbook":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"cannot load playbook {path}: {e}") from None
        rules = data.get("rules", data) if isinstance(data, dict) else data
        return cls([Rule.from_dict(r) for r in rules])

    def advise(self, turn, agent_last_line: str = "") -> str | None:
        """The coach note for the reply to `turn` (a TurnResult), or None."""
        strong = {c.name: c.strength for c in turn.cues}
        for r in self.rules:
            if r.once and r.name in self.fired:
                continue
            if turn.turn < r.turn_min:
                continue
            if r.cues and not any(strong.get(c, 0.0) >= r.min_strength for c in r.cues):
                continue
            if r.after_topic and not TOPICS[r.after_topic].search(agent_last_line or ""):
                continue
            self.fired.append(r.name)
            return r.coach
        return None
