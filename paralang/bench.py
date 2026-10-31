"""
bench.py -- does a change make the voice more expressive WITHOUT making it
harder to understand? Render fixed scripts through each engine, plain vs
expressive, then measure every render with paralang's own ears.

    doppler run -p client-acquisition-pipeline -c dev -- python -m paralang.bench
    python -m paralang.bench --engines aura,edge --out samples/expressive

Per render:
  wer          Groq Whisper word error rate vs the script (clarity; lower = better)
  f0_range_st  p90-p10 pitch in semitones over voiced frames (melody; Orpheus is the target)
  f0_move_st   median frame-to-frame pitch movement, st per 100 ms (is the contour alive?)
  dyn_db       std of 100 ms speech loudness in dB (dynamics; flat TTS is low)
  pauses       count / mean / std of pauses >= 250 ms (rhythm variety)
  dur_s        total length

These are proxies. A number going up is not "more human" by itself: the wavs
are written next to the report so a person listens before any default changes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import uuid
import wave
from pathlib import Path

import numpy as np

SCRIPTS = {
    "hook": (
             "[serious] There are four specific rocks in space, currently on a collision course with our planet. "
             "[dramatic] And none of them | are *theoretical*. One of them even has a name. Apophis. "
             "[serious] In 2004, astronomers gave it a 2.7 percent chance of hitting Earth. "
             "That number sounds small. [dramatic] It is not. For an object the size of three football fields, "
             "it is the highest risk ever recorded."),
    "agent": (
              "[warm] Hi, this is Riya from Smile Dental. [empathetic] I'm so sorry about the tooth pain, "
              "that sounds really rough. [confident] We have a slot open at four today. "
              "[curious] Would that work for you?"),
    "excited": (
                "[excited] Guess what? Your appointment is confirmed. [cheerful] We will see you tomorrow at ten. "
                "[warm] Take care until then."),
}
ORPHEUS_VOICE = {"hook": "troy", "agent": "hannah", "excited": "autumn"}
EDGE_VOICE = "en-US-AriaNeural"   # English bench; Neerja for Hinglish is unchanged
SR = 24000


def _norm_words(text: str) -> list[str]:
    t = re.sub(r"\[[a-zA-Z_:.0-9 ]+\]", " ", text).replace("|", " ").replace("*", "")
    t = t.lower().replace("2.7", "two point seven").replace("2004", "two thousand four")
    t = t.replace("%", " percent")
    return re.findall(r"[a-z']+", t)


def wer(ref: str, hyp: str) -> float:
    r, h = _norm_words(ref), _norm_words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def transcribe(path: Path) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit("GROQ_API_KEY not set (run under doppler)")
    b = "----" + uuid.uuid4().hex
    head = (f"--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode()
    body = head + path.read_bytes() + f"\r\n--{b}--\r\n".encode()
    for attempt in range(4):
        req = urllib.request.Request("https://api.groq.com/openai/v1/audio/transcriptions", data=body,
                                     headers={"Authorization": "Bearer " + key, "User-Agent": "paralang/0.3",
                                              "Content-Type": "multipart/form-data; boundary=" + b})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read()).get("text", "").strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(float(e.headers.get("retry-after") or 8))
                continue
            raise
    return ""


def measure(pcm: bytes, sr: int = SR) -> dict:
    from .features import FRAME_S, _frame_db, _pauses, _voiced_mask, pitch_track
    x = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768
    hz = pitch_track(x, sr)
    st = 12 * np.log2(hz[hz > 0] / 27.5) if (hz > 0).any() else np.zeros(0)
    out = {"dur_s": round(len(x) / sr, 2)}
    if st.size > 20:
        out["f0_range_st"] = round(float(np.percentile(st, 90) - np.percentile(st, 10)), 2)
        # movement inside voiced runs only (jumps across unvoiced gaps are not melody)
        voiced = hz > 0
        both = voiced[1:] & voiced[:-1]
        steps = np.abs(np.diff(12 * np.log2(np.maximum(hz, 1e-9) / 27.5)))[both]
        steps = steps[steps < 4]          # octave errors are tracker noise, not intonation
        out["f0_move_st"] = round(float(np.median(steps)) * 10, 2) if steps.size else None
    db = _frame_db(x, sr)
    mask = _voiced_mask(x, sr, db)
    k = int(0.1 / FRAME_S)
    blocks = [db[i:i + k].mean() for i in range(0, len(db) - k, k) if mask[i:i + k].mean() > 0.8]
    out["dyn_db"] = round(float(np.std(blocks)), 2) if len(blocks) > 3 else None
    pauses, _ = _pauses(mask)
    out["pauses"] = len(pauses)
    out["pause_mean_s"] = round(float(np.mean(pauses)), 2) if pauses else 0.0
    out["pause_std_s"] = round(float(np.std(pauses)), 2) if pauses else 0.0
    return out


def render(engine: str, name: str, text: str, expressive: bool) -> bytes:
    from .emotion import express
    from . import say
    s = express(text)
    if engine == "aura":
        return say.aura_pcm(s, expressive=expressive)
    if engine == "orpheus":
        return say.orpheus_pcm(s, ORPHEUS_VOICE[name], expressive=expressive)
    import asyncio
    return asyncio.run(say.say_pcm(text, voice=EDGE_VOICE, expressive=expressive))


def _write(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="paralang.bench")
    ap.add_argument("--engines", default="aura,edge,orpheus")
    ap.add_argument("--scripts", default=",".join(SCRIPTS))
    ap.add_argument("--out", type=Path, default=Path("samples/expressive"))
    ap.add_argument("--no-wer", action="store_true")
    a = ap.parse_args(argv)

    rows = []
    for name in a.scripts.split(","):
        text = SCRIPTS[name]
        for engine in a.engines.split(","):
            for mode in ("plain", "expressive"):
                path = a.out / f"{name}_{engine}_{mode}.wav"
                try:
                    pcm = render(engine, name, text, mode == "expressive")
                except Exception as e:  # noqa: BLE001 - one broken engine must not sink the bench
                    print(f"  ! {path.name}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                    continue
                _write(path, pcm)
                row = {"script": name, "engine": engine, "mode": mode, **measure(pcm)}
                if not a.no_wer:
                    heard = transcribe(path)
                    row["wer"] = round(wer(text, heard), 3)
                    row["heard"] = heard
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "heard"}))

    (a.out / "bench.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    cols = ["script", "engine", "mode", "wer", "f0_range_st", "f0_move_st", "dyn_db",
            "pauses", "pause_mean_s", "pause_std_s", "dur_s"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (a.out / "bench.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
