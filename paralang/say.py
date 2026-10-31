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


def _expressive(flag: bool | None) -> bool:
    """`expressive` off unless asked for, or PARALANG_EXPRESSIVE=1. It turns on
    assemble.join (trim, level match, room tone) and punctuation shaping
    (perform.shape)."""
    if flag is not None:
        return flag
    return os.environ.get("PARALANG_EXPRESSIVE", "0").strip().lower() in ("1", "true", "yes", "on")


def _finish(clips, expressive: bool) -> bytes:
    from . import assemble
    return assemble.join(clips) if expressive else assemble.raw(clips)


def _shaped(line, said: str, expressive: bool) -> str:
    if not expressive:
        return said
    from .perform import shape
    return shape(said, line.emotion, line.intensity)


async def say_pcm(text: str, emotion: str = "neutral", intensity: float = 1.0,
                  voice: str = DEFAULT_VOICE, *, expressive: bool | None = None) -> bytes:
    """24 kHz mono s16le PCM for `text` (with optional [emotion] markup)."""
    from .assemble import Clip
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH (needed to decode edge-tts mp3)")
    expressive = _expressive(expressive)
    s = express(text, emotion, intensity)
    clips = []
    for line, (sentence, kw, pause) in zip(s.lines, to_edge(s)):
        sentence = _shaped(line, sentence, expressive)
        clips.append(Clip(await _synth(sentence, edge_voice_for(sentence, voice), kw), pause))
    return _finish(clips, expressive)


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


def _edge_lines(lines, voice: str = DEFAULT_VOICE, expressive: bool = False) -> bytes:
    """Fallback voice for lines Orpheus cannot say (non-Latin script, rate limit)."""
    from .assemble import Clip
    from .emotion import Script
    s = Script(list(lines))
    clips = []
    for line, (sentence, kw, pause) in zip(s.lines, to_edge(s)):
        sentence = _shaped(line, sentence, expressive)
        clips.append(Clip(asyncio.run(_synth(sentence, edge_voice_for(sentence, voice), kw)), pause))
    return _finish(clips, expressive)


def orpheus_pcm(script, voice: str = "troy", *, fallback_voice: str = DEFAULT_VOICE,
                live: bool = False, expressive: bool | None = None) -> bytes:
    """Groq Orpheus: consecutive English lines packed into ~600-char requests
    (it acts the [directions]); Devanagari / non-Latin lines, or a chunk every
    key is rate-limited on, go to edge so no line is ever lost. Order is kept."""
    from .assemble import Clip
    from .emotion import Script
    from . import orpheus
    expressive = _expressive(expressive)

    groups: list[tuple[bool, list]] = []          # runs of (orpheus-speakable?, lines)
    for line in script.lines:
        ok = orpheus.speakable(line.text)
        if groups and groups[-1][0] == ok:
            groups[-1][1].append(line)
        else:
            groups.append((ok, [line]))

    clips = []
    for ok, lines in groups:
        pause = lines[0].prosody.pause_before_ms
        if not ok:
            # already one assembled take: keep its own edges and inner gaps
            edge_pcm = (_edge_lines(lines, fallback_voice, True) if expressive
                        else _edge_lines(lines, fallback_voice))
            clips.append(Clip(edge_pcm, pause, trim=False))
            continue
        for i, chunk in enumerate(to_orpheus(Script(lines))):
            gap = pause if i == 0 else 300                  # paragraph breath between chunks
            try:
                clips.append(Clip(orpheus.synth(chunk, voice, live=live), gap))
            except orpheus.RateLimited:
                print("  ! orpheus: every Groq key rate-limited; this chunk -> edge", file=sys.stderr)
                clips.append(Clip(asyncio.run(_synth(re.sub(r"\[[a-z_:.0-9]+\]\s*", "", chunk),
                                                     fallback_voice, {})), gap))
    return _finish(clips, expressive)


AURA_FLOW_MAX_CHARS = 1500   # Aura-2 accepts 2000; stay clear of it


def _aura_flow_groups(script, router, live: bool):
    """[(lang, [lines])]: in a batch render, consecutive English lines with the
    same emotion are spoken in ONE Aura request so the voice carries prosody
    across sentences (one request per sentence resets the contour every time
    and reads like a list). Live mode keeps one sentence per group (first audio
    must not wait for a whole paragraph)."""
    from . import aura
    groups: list[tuple[str, list]] = []
    chars = 0
    for line, (sentence, _pause) in zip(script.lines, to_aura(script)):
        lang = router.lang(sentence, reply_id=id(script))
        kind = "hi" if lang == "hi" else ("en" if aura.speakable(sentence) else "other")
        prev = groups[-1] if groups else None
        if (not live and prev and kind == "en" and prev[0] == "en"
                and prev[1][-1].emotion == line.emotion
                and chars + len(sentence) + 1 <= AURA_FLOW_MAX_CHARS):
            prev[1].append(line)
            chars += len(sentence) + 1
        else:
            groups.append((kind, [line]))
            chars = len(sentence)
    return groups


def aura_pcm(script, voice: str = "aura-2-andromeda-en", *, fallback_voice: str = DEFAULT_VOICE,
             live: bool = False, hindi_voice: str | None = None,
             expressive: bool | None = None, flow: bool = False) -> bytes:
    """Deepgram Aura-2 (Andromeda) for English, switched on the fly per sentence:
    a Hindi sentence (Devanagari or romanized, paralang.lang) is spoken by edge
    Swara, the one Hindi voice, because Andromeda garbles Hindi (samples/hindi_ab).
    Once a paragraph turns Hindi it stays Hindi until a clearly English sentence
    of SHORT_WORDS+ words. An English sentence Aura fails on -> edge Neerja.
    Planner gaps between requests, never retimed. One request per sentence.
    expressive: endings shaped (perform.shape), clips stitched by assemble.join
    with the emotion's volume applied as gain.
    flow=True (batch only, opt-in): same-emotion English runs go in one request
    (_aura_flow_groups). Measured 2026-10-31 (3 runs each, paralang.bench): flow
    NARROWED pitch range (hook 10.4 -> 7.3 st) with no clarity gain; separate
    requests give every sentence a fresh contour, which is part of what keeps
    Aura lively. So it is off, even in expressive mode."""
    from . import aura
    from .assemble import Clip, volume_db
    from .lang import HINDI_VOICE, ReplyRouter
    from .tts import _spoken
    hindi_voice = hindi_voice or HINDI_VOICE
    expressive = _expressive(expressive)
    router = ReplyRouter()
    if flow:
        groups = _aura_flow_groups(script, router, live)
    else:
        groups = []
        for line, (sentence, _p) in zip(script.lines, to_aura(script)):
            lang = router.lang(sentence, reply_id=id(script))
            groups.append(("hi" if lang == "hi" else ("en" if aura.speakable(sentence) else "other"),
                           [line]))

    clips = []
    for kind, lines in groups:
        pause = lines[0].prosody.pause_before_ms
        said = [_shaped(l, _spoken(l), expressive) for l in lines]
        text = " ".join(said)
        gain = volume_db(lines[0].prosody.volume_pct) if expressive else 0.0
        if kind == "hi":
            clips.append(Clip(asyncio.run(_synth(text, hindi_voice, lines[0].prosody.edge())), pause))
            continue
        if kind == "en":
            try:
                clips.append(Clip(aura.synth(text, voice, live=live), pause, gain))
                continue
            except aura.AuraError as e:
                print(f"  ! aura failed ({str(e)[:80]}); this sentence -> edge", file=sys.stderr)
        # fallback keeps the per-sentence planner prosody, one edge call per line
        for i, (l, s) in enumerate(zip(lines, said)):
            clips.append(Clip(asyncio.run(_synth(s, edge_voice_for(s, fallback_voice), l.prosody.edge())),
                              pause if i == 0 else l.prosody.pause_before_ms))
    return _finish(clips, expressive)


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
            voice: str | None = None, engine: str | None = None, *,
            expressive: bool | None = None) -> Path:
    engine = engine or DEFAULT_ENGINE
    voice = voice or ENGINE_VOICES[engine]
    if engine == "edge":
        pcm = asyncio.run(say_pcm(text, emotion, intensity, voice, expressive=expressive))
    elif engine == "orpheus":
        pcm = orpheus_pcm(express(text, emotion, intensity), voice, expressive=expressive)
    elif engine == "aura":
        pcm = aura_pcm(express(text, emotion, intensity), voice, expressive=expressive)
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
    ap.add_argument("--expressive", action="store_true", default=None,
                    help="trim + level-match + room tone, shaped endings "
                         "(env PARALANG_EXPRESSIVE=1)")
    a = ap.parse_args(argv)

    if a.demo:
        for name in EMOTIONS:
            p = say_wav(DEMO_LINE, Path("emotion_demo") / f"{name}.wav", name, a.intensity,
                        a.voice, a.engine, expressive=a.expressive)
            print(p)
        return
    if not a.text:
        ap.error("text is required (or use --demo)")
    if not 0 <= a.intensity <= 2:
        ap.error("--intensity must be between 0 and 2")

    s = express(a.text, a.emotion, a.intensity)
    if a.format == "wav":
        print(say_wav(a.text, a.out, a.emotion, a.intensity, a.voice, a.engine,
                      expressive=a.expressive))
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
