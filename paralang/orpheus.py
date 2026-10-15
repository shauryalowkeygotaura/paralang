"""
orpheus.py -- Groq Orpheus TTS + Groq Whisper word alignment. STDLIB ONLY.

This is paralang's default voice engine. It is deliberately one dependency-free
file so repos that cannot pip-install paralang (CI jobs: philosopher-pipeline,
football-shorts-autopilot) can vendor a copy. Edit it HERE, then re-copy.

    from paralang.orpheus import synth, stream, align, speakable
    pcm = synth("[warm] Hi there.", voice="hannah")          # 24 kHz mono s16le
    for chunk in stream("[calm] Take your time.", live=True):  # first bytes ~300-450 ms
        ...
    words = align(pcm, "Hi there.")                          # [(word, start_s, end_s)]

Measured 2026-10-15: first audio 300-470 ms per sentence (edge-tts: ~1000 ms),
whole sentence ~0.7-1.0 s. Output is WAV 24 kHz / 16-bit / mono, the same rate
paralang uses everywhere, so there is no decode step (no ffmpeg).

Limits and how they are handled (no silent quality loss):
  - English only. speakable() is False for Devanagari or other non-Latin
    script; callers route those lines to another engine (paralang: edge Neerja).
  - Free tier ~10 requests/min PER KEY. Keys rotate (GROQ_API_KEYS,
    GROQ_API_KEY, GROQ_API_KEY_2..4, same names as Scripts/groq_pool.py); a
    429 cools only that key. When every key is cooling: live=True raises
    RateLimited at once (a phone call must not stall, the caller falls back),
    live=False waits for the soonest key (batch renders never lose a line).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Iterator

SPEECH_URL = "https://api.groq.com/openai/v1/audio/speech"
STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "canopylabs/orpheus-v1-english"
ALIGN_MODEL = "whisper-large-v3-turbo"
SR = 24000
VOICES = ("autumn", "diana", "hannah", "austin", "daniel", "troy")
FEMALE = ("autumn", "diana", "hannah")
UA = "paralang/0.3"            # Groq's edge rejects Python's default User-Agent
MAX_BATCH_WAIT_S = 180         # live=False: longest total wait for a free key
NET_RETRIES = 3                # live=False: retries on a dropped connection (1 s, 2 s, 4 s)

# Latin letters, digits, punctuation, Orpheus [directions]; anything else = not speakable
_NON_LATIN = re.compile(r"[^\x00-\x7FÀ-ɏ‘-‟…–—]")


class OrpheusError(RuntimeError):
    pass


class RateLimited(OrpheusError):
    """Every key is cooling down (live mode only)."""


def speakable(text: str) -> bool:
    """True if Orpheus (English) can say this. Hindi in Devanagari, Tamil, etc. -> False."""
    return bool(text.strip()) and not _NON_LATIN.search(text)


# ── key pool ────────────────────────────────────────────────────────────────
class _Pool:
    lock = threading.Lock()
    cooling: dict[str, float] = {}     # key -> monotonic time it is usable again
    turn = 0

    @staticmethod
    def keys() -> list[str]:
        found: list[str] = []
        raw = [*(os.environ.get("GROQ_API_KEYS") or "").split(",")]
        raw += [os.environ.get(n, "") for n in ("GROQ_API_KEY", "GROQ_API_KEY_2",
                                                "GROQ_API_KEY_3", "GROQ_API_KEY_4")]
        for k in (r.strip() for r in raw):
            if k and k not in found and "paste" not in k:
                found.append(k)
        return found

    @classmethod
    def next_key(cls) -> tuple[str | None, float]:
        """(usable key, 0) or (None, seconds until the soonest key frees up)."""
        keys = cls.keys()
        if not keys:
            raise OrpheusError("no Groq key: set GROQ_API_KEY (run under `doppler run --`)")
        now = time.monotonic()
        with cls.lock:
            for i in range(len(keys)):
                k = keys[(cls.turn + i) % len(keys)]
                if cls.cooling.get(k, 0) <= now:
                    cls.turn = (cls.turn + i + 1) % len(keys)   # round-robin spreads the 10 RPM
                    return k, 0.0
            return None, max(0.05, min(cls.cooling.get(k, now) for k in keys) - now)

    @classmethod
    def cool(cls, key: str, seconds: float) -> None:
        with cls.lock:
            cls.cooling[key] = time.monotonic() + seconds


def _open(url: str, body: bytes, ctype: str, *, live: bool, timeout: float):
    """POST with key rotation. Returns an open response (caller streams it)."""
    deadline = time.monotonic() + (0 if live else MAX_BATCH_WAIT_S)
    last_err = ""
    net_fails = 0
    while True:
        key, wait = _Pool.next_key()
        if key is None:
            if live or time.monotonic() + wait > deadline:
                raise RateLimited(f"all Groq keys rate-limited ({last_err or 'cooling'})")
            time.sleep(wait)
            continue
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": ctype, "User-Agent": UA})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code == 429:
                _Pool.cool(key, float(e.headers.get("retry-after") or 7))
                last_err = "429"
                continue
            if e.code in (401, 403):
                _Pool.cool(key, 3600)          # bad/revoked key: skip it for an hour
                last_err = f"{e.code}"
                continue
            raise OrpheusError(f"Groq HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            # transient resets (WinError 10054) happen; a batch render retries with
            # backoff, a live call fails fast so the caller can fall back to edge
            net_fails += 1
            if live or net_fails > NET_RETRIES:
                raise OrpheusError(f"Groq unreachable after {net_fails} tries: {e}") from None
            time.sleep(min(8.0, 0.5 * 2 ** net_fails))


# ── speech ──────────────────────────────────────────────────────────────────
def stream(text: str, voice: str = "hannah", *, live: bool = False,
           chunk_bytes: int = 4800) -> Iterator[bytes]:
    """Yield raw PCM (24 kHz s16le mono) as it arrives. ~100 ms per chunk."""
    if voice not in VOICES:
        raise ValueError(f"unknown Orpheus voice {voice!r}; choose from {', '.join(VOICES)}")
    if not speakable(text):
        raise OrpheusError(f"not speakable by English Orpheus: {text[:40]!r}")
    body = json.dumps({"model": MODEL, "voice": voice, "input": text,
                       "response_format": "wav"}).encode("utf-8")
    with _open(SPEECH_URL, body, "application/json", live=live, timeout=60) as r:
        header = b""
        while b"data" not in header or len(header) < header.find(b"data") + 8:
            more = r.read(512)
            if not more:
                raise OrpheusError("Orpheus returned no audio")
            header += more
        start = header.find(b"data") + 8      # skip "data" + size (size is a streaming placeholder)
        carry = header[start:]
        while True:
            more = r.read(chunk_bytes)
            if not more:
                break
            carry += more
            even = len(carry) - len(carry) % 2   # never split a 16-bit sample
            if even:
                yield carry[:even]
                carry = carry[even:]
        if carry[: len(carry) - len(carry) % 2]:
            yield carry[: len(carry) - len(carry) % 2]


def synth(text: str, voice: str = "hannah", *, live: bool = False) -> bytes:
    pcm = b"".join(stream(text, voice, live=live))
    if not pcm:
        raise OrpheusError("Orpheus returned empty audio")
    return pcm


# ── alignment (captions) ────────────────────────────────────────────────────
def _wav(pcm: bytes, sr: int = SR) -> bytes:
    import struct
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
            b"data" + struct.pack("<I", len(pcm)) + pcm)


ALIGN_WINDOW_S = 240     # Groq's upload cap is ~25 MB; 240 s of 24 kHz wav is ~11.5 MB
ALIGN_SEEK_S = 6         # look this far back from a window edge for a silent cut point


def _quiet_cut(pcm: bytes, sr: int, target: int) -> int:
    """Byte offset near `target` (going back up to ALIGN_SEEK_S) at the quietest
    20 ms frame, so a window never splits a word. Stdlib only (array)."""
    import array
    frame = int(sr * 0.02) * 2
    lo = max(0, target - int(sr * ALIGN_SEEK_S) * 2)
    best, best_e = target, None
    for off in range(lo - lo % 2, target - frame, frame):
        a = array.array("h", pcm[off:off + frame])
        e = sum(x * x for x in a[::4])
        if best_e is None or e < best_e:
            best, best_e = off, e
    return best - best % 2


def _align_once(pcm: bytes, sr: int) -> list[tuple[str, float, float]]:
    b = uuid.uuid4().hex
    fields = [("model", ALIGN_MODEL), ("response_format", "verbose_json"),
              ("timestamp_granularities[]", "word"), ("language", "en")]
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                    for k, v in fields)
    body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
             f"Content-Type: audio/wav\r\n\r\n").encode() + _wav(pcm, sr) + f"\r\n--{b}--\r\n".encode()
    with _open(STT_URL, body, f"multipart/form-data; boundary={b}", live=False, timeout=120) as r:
        try:
            data = json.loads(r.read())
        except json.JSONDecodeError as e:
            raise OrpheusError(f"alignment returned bad JSON: {e}") from None
    return [(w["word"].strip(), float(w["start"]), float(w["end"]))
            for w in data.get("words", []) if w.get("word", "").strip()]


def align(pcm: bytes, text: str = "", sr: int = SR) -> list[tuple[str, float, float]]:
    """Word timings for synthesized speech via Groq Whisper.

    Long audio is aligned in ~4-minute windows cut at silences (Groq's upload
    cap), with times offset back onto the full track. Whisper's own spelling
    can drift ("Riya" -> "Ria"), so when `text` (what was submitted) has the
    same word count, the SUBMITTED words are paired with Whisper's timestamps:
    captions show the script, timed by the audio.
    """
    window = int(sr * ALIGN_WINDOW_S) * 2
    heard: list[tuple[str, float, float]] = []
    start = 0
    while start < len(pcm):
        end = len(pcm) if len(pcm) - start <= window else _quiet_cut(pcm, sr, start + window)
        if end <= start:
            end = min(len(pcm), start + window)
        offset = start / 2 / sr
        heard += [(w, a + offset, e + offset) for w, a, e in _align_once(pcm[start:end], sr)]
        start = end
    script = re.sub(r"\[[a-zA-Z_:.0-9]+\]", " ", text).split()
    if script and len(script) == len(heard):
        return [(s, a, e) for s, (_, a, e) in zip(script, heard)]
    return heard
