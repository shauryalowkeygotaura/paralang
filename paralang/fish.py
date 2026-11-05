"""
fish.py -- Fish Audio S2.1 TTS. STDLIB ONLY (vendor a copy into CI repos).

Why Fish: like Orpheus it ACTS the emotion instead of being nudged by
rate/pitch knobs, but with a much richer, free-form cue vocabulary
([excited], [slightly sad], [whispering], [laughing], [sighing], [break],
[emphasis]...) and 80+ languages. Docs read 2026-11-05:
docs.fish.audio/developer-guide/core-features/emotions.

    from paralang.fish import synth, stream, speakable
    pcm = synth("[warm] Hi there. [laughing] Ha ha, that's funny.")   # 24 kHz mono s16le

API: POST https://api.fish.audio/v1/tts, Bearer FISH_API_KEY, header
`model` (default here: s2.1-pro-free, the $0 fair-use tier; FISH_MODEL to
change, e.g. s2.1-pro). format=pcm at 24 kHz = paralang's rate, no decode.
Voice = `reference_id` (a Fish voice model id; FISH_VOICE_ID). Unset = Fish's
default voice.

Same failure contract as aura.py: live=True has a first-byte deadline and
raises FishError so the caller falls back; live=False retries 429/5xx/timeouts.
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Iterator

URL = "https://api.fish.audio/v1/tts"
SR = 24000
DEFAULT_MODEL = "s2.1-pro-free"
MODELS = ("s2.1-pro-free", "s2.1-pro", "s2-pro")    # whitelist: s1 retires 2026-12-31, uses (parens)
UA = "paralang/0.3"
LIVE_FIRST_BYTE_S = 3.0
BATCH_TIMEOUT_S = 90.0
BATCH_RETRIES = 4
MAX_CHARS = 1500            # per request; paralang chunks longer scripts
_VOICE_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class FishError(RuntimeError):
    pass


def speakable(text: str) -> bool:
    """Fish S2.1 is multilingual: anything with a letter in it."""
    return any(c.isalpha() for c in text)


def _key() -> str:
    k = (os.environ.get("FISH_API_KEY") or os.environ.get("FISH_AUDIO_API_KEY") or "").strip()
    if not k:
        raise FishError("FISH_API_KEY is not set (fish.audio -> API keys; add it to Doppler)")
    return k


def resolve_model(model: str | None = None) -> str:
    m = (model or os.environ.get("FISH_MODEL") or DEFAULT_MODEL).strip()
    if m not in MODELS:
        raise ValueError(f"unknown Fish model {m!r}; choose from {', '.join(MODELS)}")
    return m


def resolve_voice(voice: str | None = None) -> str | None:
    v = (voice or os.environ.get("FISH_VOICE_ID") or "").strip() or None
    if v is not None and not _VOICE_RE.match(v):
        raise ValueError(f"not a Fish voice id: {v!r}")
    return v


def _body(text: str, voice: str | None, speed: float, live: bool) -> bytes:
    body = {"text": text, "format": "pcm", "sample_rate": SR, "normalize": True,
            "latency": "balanced" if live else "normal"}
    if voice:
        body["reference_id"] = voice
    if abs(speed - 1.0) > 1e-3:
        body["prosody"] = {"speed": max(0.5, min(2.0, speed))}
    return json.dumps(body).encode("utf-8")


def stream(text: str, voice: str | None = None, *, model: str | None = None, live: bool = False,
           speed: float = 1.0, chunk_bytes: int = 4800) -> Iterator[bytes]:
    """Yield raw PCM (24 kHz s16le mono) as it arrives."""
    if not speakable(text):
        raise FishError(f"nothing speakable: {text[:40]!r}")
    if len(text) > MAX_CHARS:
        raise FishError(f"text is {len(text)} chars; split it (max {MAX_CHARS})")
    body = _body(text, resolve_voice(voice), speed, live)
    headers = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
               "model": resolve_model(model), "User-Agent": UA}
    timeout = LIVE_FIRST_BYTE_S if live else BATCH_TIMEOUT_S
    attempts = 1 if live else BATCH_RETRIES + 1
    for attempt in range(attempts):
        req = urllib.request.Request(URL, data=body, method="POST", headers=headers)
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (401, 403):
                raise FishError(f"Fish rejected the key (HTTP {e.code}): {detail}") from None
            if e.code == 402:
                raise FishError(f"Fish: out of credit / plan limit (HTTP 402): {detail}") from None
            if e.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(float(e.headers.get("retry-after") or 2 ** attempt))
                continue
            raise FishError(f"Fish HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as e:
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
                continue
            raise FishError(f"Fish unreachable/slow ({'live' if live else 'batch'}): {e}") from None
        carry, got = b"", False
        try:
            with r:
                while True:
                    more = r.read(chunk_bytes)
                    if not more:
                        break
                    got = True
                    carry += more
                    even = len(carry) - len(carry) % 2     # never split a 16-bit sample
                    if even:
                        yield carry[:even]
                        carry = carry[even:]
        except (TimeoutError, socket.timeout, ConnectionError, OSError) as e:
            if not got and attempt < attempts - 1:
                time.sleep(2 ** attempt)
                continue
            raise FishError(f"Fish stream broke: {e}") from None
        if not got:
            raise FishError("Fish returned no audio")
        return
    raise FishError("Fish: retries exhausted")


def synth(text: str, voice: str | None = None, *, model: str | None = None, live: bool = False,
          speed: float = 1.0) -> bytes:
    return b"".join(stream(text, voice, model=model, live=live, speed=speed))
