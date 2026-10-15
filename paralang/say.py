"""
say.py -- emotional speech from the command line or Python, with the free
edge-tts voices. For reels, voiceovers, demos: anything, not just agents.

    python -m paralang.say "[warm] Namaste ji. [excited] Aaj bada offer hai!" -o hi.wav
    python -m paralang.say "Socho... agar sab kuch ek sapna ho?" -e dramatic -v en-IN-PrabhatNeural -o reel.wav
    python -m paralang.say --demo           # every emotion, same line, into ./emotion_demo/
    python -m paralang.say "..." -e warm --format ssml|elevenlabs|openai|gemini   # just print the payload
    python -m paralang.say "[dramatic] And none of them | are *theoretical*." --engine orpheus -v troy
    python -m paralang.say "..." --engine aura -v aura-2-andromeda-en       # needs DEEPGRAM_API_KEY

Engines (default: aura; override with --engine or PARALANG_ENGINE):
  aura     Deepgram Aura-2, voice aura-2-andromeda-en, DEEPGRAM_API_KEY. No emotion
           control: paced by sentence gaps and "..." beats. Devanagari lines go to
           edge Swara (hi-IN); a sentence Aura fails on goes to edge Neerja.
  orpheus  Groq canopylabs/orpheus-v1-english, GROQ_API_KEY(S). ACTS the emotion;
           voices autumn diana hannah austin daniel troy. Same fallbacks.
  edge     free, knob-based emotion (rate/pitch/volume), any language.

    from paralang.say import say_wav
    say_wav("Take a breath.", "out.wav")                              # aura andromeda
    say_wav("[calm] Saans lo.", "out.wav", engine="edge", voice="en-IN-NeerjaNeural")
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
import shutil
import subprocess
import sys
import wave
from pathlib import Path

from .emotion import EMOTIONS, express
from .tts import to_aura, to_edge, to_elevenlabs, to_gemini, to_openai, to_orpheus, to_ssml

DEFAULT_VOICE = "en-IN-NeerjaNeural"
SR = 24000
DEMO_LINE = "Achha... toh aapko lagta hai ye kaam karega? Main bata deti hoon, ye sach mein badal sakta hai."


async def _synth(text: str, voice: str, kw: dict) -> bytes:
    try:
        import edge_tts
    except ImportError:
        raise SystemExit("edge-tts not installed: pip install 'paralang[say]'")
    mp3 = bytearray()
    async for c in edge_tts.Communicate(text, voice, **kw).stream():
        if c.get("type") == "audio":
            mp3.extend(c["data"])
    if not mp3:
        raise RuntimeError(f"edge-tts returned no audio for {text[:40]!r} (voice {voice})")
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1",
                        "-ar", str(SR), "pipe:1"], input=bytes(mp3), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {r.stderr.decode(errors='replace')[:200]}")
    return r.stdout


async def say_pcm(text: str, emotion: str = "neutral", intensity: float = 1.0,
                  voice: str = DEFAULT_VOICE) -> bytes:
    """24 kHz mono s16le PCM for `text` (with optional [emotion] markup)."""
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH (needed to decode edge-tts mp3)")
    pcm = bytearray()
    for sentence, kw, pause in to_edge(express(text, emotion, intensity)):
        pcm += b"\x00\x00" * (SR * pause // 1000)
        pcm += await _synth(sentence, edge_voice_for(sentence, voice), kw)
    return bytes(pcm)


def _decode(audio: bytes) -> bytes:
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1",
                        "-ar", str(SR), "pipe:1"], input=audio, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {r.stderr.decode(errors='replace')[:200]}")
    return r.stdout


def _http(url: str, payload: dict, auth: str, retries: int = 4) -> bytes:
    """POST JSON, return audio bytes. Waits out a 429 (the free tier is 10/min)."""
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": auth, "Content-Type": "application/json",
            "User-Agent": "paralang/0.2"})  # Groq's edge rejects Python's default UA
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            if not data:
                raise RuntimeError(f"{url}: empty audio")
            return data
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code == 429 and attempt < retries - 1:
                time.sleep(float(e.headers.get("retry-after") or 7))
                continue
            raise RuntimeError(f"{url} -> HTTP {e.code}: {detail}") from None
    raise RuntimeError(f"{url}: still rate-limited after {retries} tries")


def _key(name: str) -> str:
    k = os.environ.get(name, "").strip()
    if not k:
        raise SystemExit(f"{name} is not set (run under `doppler run --`)")
    return k


# edge's en-IN voices return NO audio for Devanagari (measured 2026-10-15), so a
# Hindi-script line must go to a hi-IN voice. Same gender as Neerja by default.
SCRIPT_VOICES = [(re.compile(r"[ऀ-ॿ]"), "hi-IN-SwaraNeural")]


def edge_voice_for(text: str, default: str = DEFAULT_VOICE) -> str:
    """The edge voice that can actually speak `text`'s script."""
    for pattern, voice in SCRIPT_VOICES:
        if pattern.search(text):
            return voice
    return default


def _edge_lines(lines, voice: str = DEFAULT_VOICE) -> bytes:
    """Fallback voice for lines Orpheus cannot say (non-Latin script, rate limit)."""
    from .emotion import Script
    pcm = bytearray()
    for sentence, kw, pause in to_edge(Script(list(lines))):
        pcm += b"\x00\x00" * (SR * pause // 1000)
        pcm += asyncio.run(_synth(sentence, edge_voice_for(sentence, voice), kw))
    return bytes(pcm)


def orpheus_pcm(script, voice: str = "troy", *, fallback_voice: str = DEFAULT_VOICE,
                live: bool = False) -> bytes:
    """Groq Orpheus: consecutive English lines packed into ~600-char requests
    (it acts the [directions]); Devanagari / non-Latin lines, or a chunk every
    key is rate-limited on, go to edge so no line is ever lost. Order is kept."""
    from .emotion import Script
    from . import orpheus

    groups: list[tuple[bool, list]] = []          # runs of (orpheus-speakable?, lines)
    for line in script.lines:
        ok = orpheus.speakable(line.text)
        if groups and groups[-1][0] == ok:
            groups[-1][1].append(line)
        else:
            groups.append((ok, [line]))

    pcm = bytearray()
    for ok, lines in groups:
        if pcm:
            pcm += b"\x00\x00" * (SR * lines[0].prosody.pause_before_ms // 1000)
        if not ok:
            pcm += _edge_lines(lines, fallback_voice)
            continue
        sub = Script(lines)
        for i, chunk in enumerate(to_orpheus(sub)):
            if i:
                pcm += b"\x00\x00" * (SR * 300 // 1000)   # paragraph breath between chunks
            try:
                pcm += orpheus.synth(chunk, voice, live=live)
            except orpheus.RateLimited:
                print("  ! orpheus: every Groq key rate-limited; this chunk -> edge", file=sys.stderr)
                pcm += asyncio.run(_synth(re.sub(r"\[[a-z_:.0-9]+\]\s*", "", chunk),
                                          fallback_voice, {}))
    return bytes(pcm)


def aura_pcm(script, voice: str = "aura-2-andromeda-en", *, fallback_voice: str = DEFAULT_VOICE,
             live: bool = False) -> bytes:
    """Deepgram Aura-2 (Andromeda): one request per sentence, planner gaps
    between, never retimed. A sentence Aura cannot say goes to edge so nothing
    is dropped: Devanagari -> Swara (hi-IN), any Aura failure -> Neerja, both
    with the planner's rate/pitch/volume knobs."""
    from . import aura
    pcm = bytearray()
    for line, (sentence, pause) in zip(script.lines, to_aura(script)):
        pcm += b"\x00\x00" * (SR * pause // 1000)
        if aura.speakable(sentence):
            try:
                pcm += aura.synth(sentence, voice, live=live)
                continue
            except aura.AuraError as e:
                print(f"  ! aura failed ({str(e)[:80]}); this sentence -> edge", file=sys.stderr)
        pcm += asyncio.run(_synth(sentence, edge_voice_for(sentence, fallback_voice),
                                  line.prosody.edge()))
    return bytes(pcm)


ENGINE_VOICES = {"edge": DEFAULT_VOICE, "orpheus": "troy", "aura": "aura-2-andromeda-en"}
# Default = Deepgram Aura-2 Andromeda (Shaurya, 2026-10-15). Fallbacks: edge Neerja,
# and hi-IN Swara for Devanagari. PARALANG_ENGINE=orpheus|edge overrides.
DEFAULT_ENGINE = (os.environ.get("PARALANG_ENGINE") or "aura").strip().lower()
if DEFAULT_ENGINE not in ENGINE_VOICES:
    # warn, never kill: agents import this module and must keep running
    print(f"paralang: PARALANG_ENGINE={DEFAULT_ENGINE!r} unknown (choose {', '.join(ENGINE_VOICES)}); "
          "using aura", file=sys.stderr)
    DEFAULT_ENGINE = "aura"


def say_wav(text: str, out: str | Path, emotion: str = "neutral", intensity: float = 1.0,
            voice: str | None = None, engine: str | None = None) -> Path:
    engine = engine or DEFAULT_ENGINE
    voice = voice or ENGINE_VOICES[engine]
    if engine == "edge":
        pcm = asyncio.run(say_pcm(text, emotion, intensity, voice))
    elif engine == "orpheus":
        pcm = orpheus_pcm(express(text, emotion, intensity), voice)
    elif engine == "aura":
        pcm = aura_pcm(express(text, emotion, intensity), voice)
    else:
        raise ValueError(f"unknown engine {engine!r}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="paralang.say", description="emotional TTS for any voice")
    ap.add_argument("text", nargs="?", help="text; may contain [emotion] or [emotion:0.5] markup")
    ap.add_argument("-e", "--emotion", default="neutral", choices=sorted(EMOTIONS))
    ap.add_argument("-i", "--intensity", type=float, default=1.0, help="0..2 (default 1)")
    ap.add_argument("-v", "--voice", default=None, help="voice name for the chosen engine")
    ap.add_argument("--engine", choices=sorted(ENGINE_VOICES), default=DEFAULT_ENGINE,
                    help=f"default {DEFAULT_ENGINE} (env PARALANG_ENGINE)")
    ap.add_argument("-o", "--out", type=Path, default=Path("say.wav"))
    ap.add_argument("--format", choices=["wav", "ssml", "azure", "elevenlabs", "openai", "gemini"],
                    default="wav", help="wav = synthesize with --engine; others print the payload")
    ap.add_argument("--demo", action="store_true", help="render every emotion to ./emotion_demo/")
    a = ap.parse_args(argv)

    if a.demo:
        for name in EMOTIONS:
            p = say_wav(DEMO_LINE, Path("emotion_demo") / f"{name}.wav", name, a.intensity,
                        a.voice, a.engine)
            print(p)
        return
    if not a.text:
        ap.error("text is required (or use --demo)")
    if not 0 <= a.intensity <= 2:
        ap.error("--intensity must be between 0 and 2")

    s = express(a.text, a.emotion, a.intensity)
    if a.format == "wav":
        print(say_wav(a.text, a.out, a.emotion, a.intensity, a.voice, a.engine))
    elif a.format in ("ssml", "azure"):
        print(to_ssml(s, a.voice or DEFAULT_VOICE, azure=a.format == "azure"))
    elif a.format == "elevenlabs":
        print(json.dumps(to_elevenlabs(s), ensure_ascii=False, indent=2))
    elif a.format == "openai":
        print(json.dumps(to_openai(s), ensure_ascii=False, indent=2))
    else:
        print(to_gemini(s))


if __name__ == "__main__":
    sys.exit(main())
