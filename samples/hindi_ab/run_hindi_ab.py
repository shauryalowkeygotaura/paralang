"""Is Andromeda usable for Hindi? Objective A/B, plus wavs to listen to.

Each line is spoken by every candidate voice, then Groq Whisper (language=hi)
transcribes it back. Score = character error rate of that Devanagari transcript
against the reference Devanagari line (spaces/punctuation ignored). Lower =
clearer Hindi. Andromeda / Neerja / Orpheus get the ROMANIZED line (they cannot
read Devanagari); Swara gets the Devanagari line.

    doppler run --project client-acquisition-pipeline --config dev -- \
        ../../../jio-outbound/local_agent/.venv-rt/Scripts/python run_hindi_ab.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paralang import aura, orpheus          # noqa: E402
from paralang.say import SR, _synth          # noqa: E402

OUT = Path(__file__).parent
LINES = [
    ("Aapka plan kal khatam ho raha hai, kya main aapko naya plan bataun?",
     "आपका प्लान कल खत्म हो रहा है, क्या मैं आपको नया प्लान बताऊं?"),
    ("Koi baat nahi ji, aap aaram se sochiye.",
     "कोई बात नहीं जी, आप आराम से सोचिए।"),
    ("Main aapko WhatsApp pe details bhej deti hoon.",
     "मैं आपको व्हाट्सएप पे डिटेल्स भेज देती हूं।"),
    ("Kal subah das baje doctor sahab free hain.",
     "कल सुबह दस बजे डॉक्टर साहब फ्री हैं।"),
]


def voices():
    return {
        "andromeda (roman)": lambda roman, deva: aura.synth(roman),
        "neerja (roman)": lambda roman, deva: asyncio.run(_synth(roman, "en-IN-NeerjaNeural", {})),
        "swara (devanagari)": lambda roman, deva: asyncio.run(_synth(deva, "hi-IN-SwaraNeural", {})),
        "swara (roman)": lambda roman, deva: asyncio.run(_synth(roman, "hi-IN-SwaraNeural", {})),
    }


def norm(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[len(h)] / max(1, len(r))


def transcribe_hi(pcm: bytes) -> str:
    import uuid
    import urllib.request
    b = uuid.uuid4().hex
    fields = [("model", "whisper-large-v3-turbo"), ("language", "hi"), ("response_format", "json")]
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                    for k, v in fields)
    body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
             f"Content-Type: audio/wav\r\n\r\n").encode() + orpheus._wav(pcm) + f"\r\n--{b}--\r\n".encode()
    with orpheus._open(orpheus.STT_URL, body, f"multipart/form-data; boundary={b}",
                       live=False, timeout=120) as r:
        return json.loads(r.read()).get("text", "").strip()


def save(name: str, pcm: bytes) -> None:
    with wave.open(str(OUT / name), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm)


def main() -> None:
    rows, totals = [], {}
    for i, (roman, deva) in enumerate(LINES, 1):
        for name, speak in voices().items():
            t = time.perf_counter()
            try:
                pcm = speak(roman, deva)
            except Exception as e:  # noqa: BLE001
                rows.append((i, name, None, f"SYNTH FAILED: {e}"[:80]))
                continue
            took = time.perf_counter() - t
            save(f"line{i}_{name.split()[0]}_{name.split()[1].strip('()')}.wav", pcm)
            heard = transcribe_hi(pcm)
            c = cer(deva, heard)
            totals.setdefault(name, []).append(c)
            rows.append((i, name, c, f"{took:4.1f}s  heard: {heard}"))
            print(f"line {i} | {name:20s} | CER {c:5.1%} | {took:4.1f}s | {heard}", flush=True)
    print("\nMEAN CER (lower = clearer Hindi):")
    for name, cs in sorted(totals.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {name:20s} {sum(cs) / len(cs):6.1%}  over {len(cs)} lines")
    (OUT / "results.json").write_text(json.dumps(
        {"lines": LINES, "rows": rows, "mean_cer": {k: sum(v) / len(v) for k, v in totals.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
