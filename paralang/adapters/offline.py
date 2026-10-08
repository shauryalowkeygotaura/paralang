"""
Offline adapter: a recorded call -> one JSON line per caller turn.

For hosted agents (VAPI, Retell) that never hand you live audio: download the
recording after the call and run this. Useful for scoring, QA and the weekly
retainer review ("which calls did patients sound unsure on?").

    python -m paralang.adapters.offline call.wav                       # whole file = one turn
    python -m paralang.adapters.offline call.wav --text "haan theek hai"
    python -m paralang.adapters.offline call.wav --turns turns.json --channel 0

turns.json: [{"start": 3.2, "end": 5.9, "text": "...", "response_latency_s": 0.8}, ...]
            (seconds into the file; VAPI's transcript messages carry secondsFromStart)
--channel:  pick the caller's channel from a stereo recording (VAPI stereo puts
            each party on its own channel); default mixes down.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

from ..analyzer import Analyzer


def read_wav(path: Path, channel: int | None) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError, OSError) as e:
        raise SystemExit(f"cannot read {path} as PCM wav: {e} (convert with: ffmpeg -i in -ac 1 -ar 16000 out.wav)")
    if sr <= 0 or ch <= 0:
        raise SystemExit(f"{path}: corrupt wav header (rate={sr}, channels={ch})")
    if width != 2:
        raise SystemExit(f"{path}: {8 * width}-bit wav; need 16-bit PCM (ffmpeg -i in -sample_fmt s16 out.wav)")
    x = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        x = x[: len(x) - len(x) % ch].reshape(-1, ch)
        if channel is not None:
            if not 0 <= channel < ch:
                raise SystemExit(f"--channel {channel} out of range: file has {ch} channels")
            x = x[:, channel]
        else:
            x = x.mean(axis=1).astype(np.int16)
    return x, sr


def load_turns(path: Path) -> list[dict]:
    try:
        turns = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read turns file {path}: {e}")
    if not isinstance(turns, list):
        raise SystemExit("turns file must be a JSON list of {start, end, text}")
    out = []
    for i, t in enumerate(turns):
        try:
            start, end = float(t["start"]), float(t["end"])
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"turn {i}: needs numeric 'start' and 'end' (seconds)")
        if end <= start:
            raise SystemExit(f"turn {i}: end ({end}) must be after start ({start})")
        lat = t.get("response_latency_s")
        try:
            lat = float(lat) if lat is not None else None
        except (TypeError, ValueError):
            raise SystemExit(f"turn {i}: response_latency_s must be a number, got {lat!r}")
        out.append({"start": start, "end": end, "text": str(t.get("text", "")),
                    "response_latency_s": lat})
    return out


def analyze_file(path: Path, turns: list[dict] | None = None, text: str = "",
                 channel: int | None = None, analyzer: Analyzer | None = None) -> list[dict]:
    x, sr = read_wav(path, channel)
    an = analyzer or Analyzer()
    turns = turns or [{"start": 0.0, "end": len(x) / sr, "text": text, "response_latency_s": None}]
    results = []
    for t in turns:
        seg = x[int(t["start"] * sr): int(t["end"] * sr)]
        if seg.size == 0:
            print(f"warn: turn {t['start']}-{t['end']}s is past the end of the "
                  f"{len(x) / sr:.1f}s recording; skipped", file=sys.stderr)
            continue
        res = an.analyze_turn(seg, sr, t["text"], response_latency_s=t["response_latency_s"])
        results.append(res.to_dict())
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="paralang-offline", description=__doc__.split("\n\n")[0])
    ap.add_argument("wav", type=Path)
    ap.add_argument("--turns", type=Path, help="JSON list of caller turns")
    ap.add_argument("--text", default="", help="transcript when the whole file is one turn")
    ap.add_argument("--channel", type=int, help="caller channel in a stereo file")
    ap.add_argument("--out", type=Path, help="write JSONL here instead of stdout")
    a = ap.parse_args(argv)

    rows = analyze_file(a.wav, load_turns(a.turns) if a.turns else None, a.text, a.channel)
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if a.out:
        a.out.write_text(lines + "\n", encoding="utf-8")
        print(f"{len(rows)} turns -> {a.out}", file=sys.stderr)
    else:
        print(lines)


if __name__ == "__main__":
    main()
