"""
aura.py -- Deepgram Aura-2 TTS (default voice: Andromeda). STDLIB ONLY.

paralang's default engine since 2026-10-15 (Shaurya: "Andromeda everything").
One dependency-free file, so CI repos can vendor a copy. Edit it HERE.

    from paralang.aura import synth, stream, speakable
    pcm = synth("Namaste ji, main Riya bol rahi hoon.")      # 24 kHz mono s16le
    for chunk in stream("Take your time.", live=True): ...

Deepgram returns raw linear16 at 24 kHz (container=none), the same rate
paralang uses everywhere: no decode step, no ffmpeg.

Measured 2026-10-15 from Delhi: first audio 1.4-2.5 s, whole sentence 3-10 s,
one read timed out. That is why:
  - live=True has a FIRST-BYTE deadline (LIVE_FIRST_BYTE_S): a phone caller
    must never sit in dead air, so a slow sentence raises AuraError and the
    caller speaks it with the fallback voice (paralang: edge Neerja).
  - live=False retries timeouts, dropped connections, 429 and 5xx with backoff.
Aura has NO emotion control: paralang paces it with sentence gaps and "..."
beats only, and never retimes fragments (2026-10-08 A/B: that hurt clarity).
It is English-only, like Orpheus: speakable() routes other scripts away.
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator

URL = "https://api.deepgram.com/v1/speak"
SR = 24000
DEFAULT_VOICE = "aura-2-andromeda-en"
UA = "paralang/0.3"
LIVE_FIRST_BYTE_S = 3.0     # live: no audio by then -> the sentence goes to the fallback voice
BATCH_TIMEOUT_S = 45.0
BATCH_RETRIES = 4           # backoff 1, 2, 4, 8 s

_NON_LATIN = re.compile(r"[^\x00-\x7FÀ-ɏ‘-‟…–—]")
_VOICE_RE = re.compile(r"^aura(-2)?-[a-z]+-[a-z]{2}$")   # whitelist, e.g. aura-2-andromeda-en


class AuraError(RuntimeError):
    pass


def speakable(text: str) -> bool:
    """True if Aura (English) can say this. Devanagari etc. -> False."""
    return bool(text.strip()) and not _NON_LATIN.search(text)


def _key() -> str:
    k = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not k:
        raise AuraError("DEEPGRAM_API_KEY is not set (Doppler dental-receptionist/dev has it)")
    return k


def stream(text: str, voice: str = DEFAULT_VOICE, *, live: bool = False,
           chunk_bytes: int = 4800) -> Iterator[bytes]:
    """Yield raw PCM (24 kHz s16le mono) as it arrives (~100 ms per chunk)."""
    if not _VOICE_RE.match(voice):
        raise ValueError(f"not an Aura voice id: {voice!r} (e.g. {DEFAULT_VOICE})")
    if not speakable(text):
        raise AuraError(f"not speakable by English Aura: {text[:40]!r}")
    q = urllib.parse.urlencode({"model": voice, "encoding": "linear16",
                                "sample_rate": SR, "container": "none"})
    body = json.dumps({"text": text}).encode("utf-8")
    attempts = 1 if live else BATCH_RETRIES + 1
    for attempt in range(attempts):
        req = urllib.request.Request(f"{URL}?{q}", data=body, method="POST", headers={
            "Authorization": f"Token {_key()}", "Content-Type": "application/json", "User-Agent": UA})
        try:
            r = urllib.request.urlopen(req, timeout=LIVE_FIRST_BYTE_S if live else BATCH_TIMEOUT_S)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:200]
            if e.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(float(e.headers.get("retry-after") or 2 ** attempt))
                continue
            raise AuraError(f"Deepgram HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as e:
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
                continue
            raise AuraError(f"Deepgram unreachable/slow ({'live' if live else 'batch'}): {e}") from None
        with r:
            # urllib's timeout applies per socket read, so in live mode every gap
            # (first audio included) is capped at LIVE_FIRST_BYTE_S
            carry, got = b"", False
            try:
                while True:
                    more = r.read(chunk_bytes)
                    if not more:
                        break
                    got = True
                    carry += more
                    even = len(carry) - len(carry) % 2
                    if even:
                        yield carry[:even]
                        carry = carry[even:]
            except (TimeoutError, socket.timeout, ConnectionError) as e:
                if not got and attempt < attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise AuraError(f"Deepgram stream broke: {e}") from None
            if not got:
                raise AuraError("Deepgram returned no audio")
            return
    raise AuraError("Deepgram: retries exhausted")


def synth(text: str, voice: str = DEFAULT_VOICE, *, live: bool = False) -> bytes:
    return b"".join(stream(text, voice, live=live))
