"""A/B for format-render voice pacing. Same hook, four deliveries.

    doppler run -p dental-receptionist -c dev -- python make_ab.py        (Aura: needs DEEPGRAM_API_KEY)
    doppler run -p philosopher-pipeline -c dev -- python make_ab.py --orpheus

Markup used by B and C (proposed paralang syntax):
    [emotion]  switch emotion for the following sentences
    |          a beat: a short dramatic pause inside a sentence
    *word*     emphasis: a beat before it, and it is read slower
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent
PLAIN = ("There are four specific rocks in space, currently on a collision course with our planet. "
         "And none of them are theoretical. One of them even has a name. Apophis.")
MARKED = ("[serious] There are four specific rocks in space, currently on a collision course with our planet. "
          "[dramatic] And none of them | are *theoretical*. [serious] One of them even has a name. | *Apophis*.")
UA = {"User-Agent": "curl/8"}
SR = 24000


def _post(url, body, headers, timeout=60):
    req = urllib.request.Request(url, data=body, headers={**headers, **UA}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{url} -> HTTP {e.code}: {e.read().decode()[:300]}")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def to_wav(src_bytes, ext, dst, tempo=1.0):
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(src_bytes)
    af = ["-af", f"atempo={tempo:.3f}"] if abs(tempo - 1) > 0.005 else []
    ffmpeg("-i", f.name, *af, "-ar", str(SR), "-ac", "1", str(dst))
    os.unlink(f.name)


def silence(ms, dst):
    ffmpeg("-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=mono", "-t", f"{ms / 1000:.3f}", str(dst))


def concat(parts, dst):
    lst = Path(tempfile.mkdtemp()) / "list.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(dst))


# Same numbers paralang uses (EMOTIONS rate %, pause multiplier) for the two emotions here.
EMO = {"serious": (-5, 1.2), "dramatic": (-12, 1.6), "neutral": (0, 1.0)}
SENT_GAP_MS = 320
BEAT_MS = 420
EMPH_TEMPO = 0.88


def segments(marked):
    """-> list of (kind, text, emotion): kind is 'say', 'beat' or 'gap'."""
    out, emo = [], "neutral"
    for tok in re.split(r"(\[[a-z]+\])", marked):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"\[([a-z]+)\]", tok)
        if m:
            emo = m.group(1)
            continue
        for sent in re.findall(r"[^.!?]+[.!?]*", tok):
            sent = sent.strip()
            if not sent:
                continue
            if out:
                out.append(("gap", "", emo))
            pieces = re.split(r"\s*\|\s*", sent)
            for i, piece in enumerate(pieces):
                if i:
                    out.append(("beat", "", emo))
                for j, sub in enumerate(re.split(r"(\*[^*]+\*)", piece)):
                    sub = sub.strip()
                    if not sub:
                        continue
                    # Bare punctuation joins the previous spoken piece: sent alone,
                    # a TTS may read "." as "dot" or produce a click.
                    if not any(ch.isalnum() for ch in sub) and out and out[-1][0] in ("say", "emph"):
                        k, t, e = out[-1]
                        out[-1] = (k, t + sub, e)
                        continue
                    if sub.startswith("*"):
                        out.append(("emph", sub.strip("*"), emo))
                    else:
                        out.append(("say", sub, emo))
    return out


def aura(text):
    key = os.environ["DEEPGRAM_API_KEY"].strip()
    return _post("https://api.deepgram.com/v1/speak?model=aura-2-andromeda-en&encoding=mp3",
                 json.dumps({"text": text}).encode(),
                 {"Authorization": f"Token {key}", "Content-Type": "application/json"})


_last = [0.0]


def orpheus(text, voice):
    # Free tier: 10 requests/minute. Space calls out instead of failing.
    import time
    wait = 6.2 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    key = os.environ["GROQ_API_KEY"].strip()
    return _post("https://api.groq.com/openai/v1/audio/speech",
                 json.dumps({"model": "canopylabs/orpheus-v1-english", "voice": voice,
                             "input": text, "response_format": "wav"}).encode(),
                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})


def paced(render, ext, marked, dst, use_tags=False):
    tmp = Path(tempfile.mkdtemp())
    parts = []
    for n, (kind, text, emo) in enumerate(segments(marked)):
        p = tmp / f"{n:03d}.wav"
        rate, pause_mul = EMO.get(emo, (0, 1.0))
        if kind == "gap":
            silence(SENT_GAP_MS * pause_mul, p)
        elif kind == "beat":
            silence(BEAT_MS * pause_mul, p)
        else:
            say = f"[{emo}] {text}" if use_tags and emo != "neutral" else text
            tempo = (1 + rate / 100) * (EMPH_TEMPO if kind == "emph" else 1)
            # Orpheus acts the emotion itself, so only emphasis is retimed there.
            if use_tags:
                tempo = EMPH_TEMPO if kind == "emph" else 1.0
            if kind == "emph":
                pb = tmp / f"{n:03d}b.wav"
                silence(160, pb)
                parts.append(pb)
            to_wav(render(say), ext, p, tempo)
        parts.append(p)
    concat(parts, dst)


if __name__ == "__main__":
    if "--orpheus" in sys.argv:
        for v in ("hannah", "troy"):
            to_wav(orpheus(PLAIN, v), ".wav", OUT / f"C0_orpheus_{v}_flat.wav")
            paced(lambda t, v=v: orpheus(t, v), ".wav", MARKED, OUT / f"C_orpheus_{v}_paced.wav", use_tags=True)
        print("orpheus samples written")
    else:
        to_wav(aura(PLAIN), ".mp3", OUT / "A_aura_flat.wav")
        paced(aura, ".mp3", MARKED, OUT / "B_aura_paralang_paced.wav")
        print("aura samples written")
