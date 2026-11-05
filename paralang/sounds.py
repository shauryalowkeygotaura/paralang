"""
sounds.py -- a bank of real nonverbal sounds (laugh, sigh, gasp, breath...)
for engines that cannot make them.

Fish and Orpheus ACT "[laughing]"; Aura and edge would read it aloud (so
paralang strips it) or skip it. Here the sound is performed once by Orpheus
(which can laugh), cached on disk, and spliced into the Aura/edge render at the
cue's position. A gender-matched Orpheus voice performs it so a female voice
does not suddenly laugh in a man's voice.

    from paralang.sounds import get
    pcm = get("laugh", voice="aura-2-andromeda-en", seed="line text")   # or None

    python -m paralang.sounds --build            # pre-render the whole bank (Groq key)

Quality gate (measured 2026-11-05): each clip is transcribed by Groq Whisper;
a clip in which Whisper hears a real word (not "ha", "hmm", "ah"...) is a voice
SAYING the sound, not making it, and is rejected. Clips are trimmed and capped
at MAX_S per sound. Nothing works -> get() returns None and the cue is skipped:
a missing laugh is better than a wrong one.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np

SR = 24000
# sound -> (Orpheus prompts, one per variant; max seconds)
BANK = {
    "laugh":   (["[laughing] Hahaha!", "[laughs] Haha.", "[laughs warmly] Hahah."], 1.6),
    "chuckle": (["[chuckles] Heh.", "[chuckling softly] Hm hm.", "[chuckles] Heh heh."], 1.0),
    "sigh":    (["[sighs] Hhhh.", "[sighing] Haah...", "[sighs softly] Hhh.", "[sighs deeply] Haaaah.",
                 "[exhales slowly] Hhhhhh."], 1.3),
    "gasp":    (["[gasps] Oh!", "[gasps] Ah!"], 0.7),
    "breath":  (["[breathes in] Hhh.", "[inhales] Hh.", "[takes a deep breath] Hhhhh.",
                 "[inhales sharply] Hh!"], 0.7),
    "yawn":    (["[yawning] Aaahh.", "[yawns] Hhaaa.", "[yawns loudly] Haaaaahh.", "[sleepy yawn] Aaaawh."], 1.6),
    "sob":     (["[sobbing] Hhh hh.", "[crying softly] Hh."], 1.6),
    "groan":   (["[groans] Uuugh.", "[groaning] Ugh."], 1.0),
    "throat":  (["[clears throat] Ahem.", "[clears throat] Hm-hm."], 0.8),
    "hmm":     (["[thoughtful] Hmm...", "[thinking] Hmmm."], 1.0),
}
# what Whisper may "hear" in a real nonverbal clip (it hallucinates "you" on non-speech)
ONOMATOPOEIA = set("ha haha hahaha hah heh hm hmm hmmm mm mmm hmph mhm hmhm ah aah ahh oh ooh "
                   "uh uh-huh huh ugh uugh bah ahem you yeah oof whew phew aw ow hhh h mm-hmm".split())
MALE = {"troy", "austin", "daniel"}
MALE_HINTS = ("prabhat", "orion", "arcas", "apollo", "zeus", "atlas", "orpheus", "hermes", "mars",
              "neptune", "odysseus", "jupiter", "aries", "draco", "hyperion", "guy", "davis", "andrew",
              "brian", "christopher", "eric", "roger", "steffan", "madhur", "male")
FEMALE_PERFORMER, MALE_PERFORMER = "hannah", "troy"


def performer_for(voice: str | None) -> str:
    """Orpheus voice that performs sounds for `voice` (same gender; default female)."""
    v = (voice or "").lower()
    if v in MALE or any(h in v for h in MALE_HINTS):
        return MALE_PERFORMER
    return FEMALE_PERFORMER


def cache_dir() -> Path:
    base = os.environ.get("PARALANG_SOUNDS_DIR")
    return Path(base) if base else Path.home() / ".cache" / "paralang" / "sounds"


def _path(sound: str, performer: str, i: int) -> Path:
    return cache_dir() / performer / f"{sound}_{i}.pcm"


def _rejected(sound: str, performer: str, i: int) -> Path:
    return cache_dir() / performer / f"{sound}_{i}.rejected"


def heard_words_ok(heard: str) -> bool:
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", heard.lower())
    # drawn-out sounds Whisper spells freely: "heheheheh", "awwww", "hmmmm", "aaahh"
    return all(w in ONOMATOPOEIA or _STRETCHED.match(w) for w in words)


# Only interjection shapes, no nested quantifiers: "hehehe(h)", "hahaha",
# "awww", "hmmmm", "aaahh", "ooohh". Short real words ("i", "hi", "am") never match.
_STRETCHED = re.compile(r"^(?:he){2,}h?$|^(?:ha){2,}h?$|^a+w{2,}$|^h?m{3,}$|^a{2,}h*$|^o{2,}h*$")


def _shape(pcm: bytes, max_s: float) -> bytes:
    from .assemble import _f32, trim
    x = trim(_f32(pcm))
    n = int(SR * max_s)
    if len(x) > n:
        x = x[:n].copy()
        k = int(SR * 0.12)                      # fade the cut, a sound must not stop dead
        x[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def _whisper(pcm: bytes) -> str | None:
    """Transcript of a clip, or None if Whisper is unavailable (gate skipped)."""
    try:
        from . import orpheus
        return " ".join(w for w, _, _ in orpheus.align(pcm))
    except Exception as e:  # noqa: BLE001 - the gate is best effort
        print(f"  ! sounds: whisper gate unavailable ({str(e)[:60]})", file=sys.stderr)
        return None


def build_one(sound: str, performer: str, i: int, *, live: bool = False) -> bytes | None:
    """Render, gate and cache variant i. None if it failed or was rejected."""
    from . import orpheus
    prompts, max_s = BANK[sound]
    try:
        raw = orpheus.synth(prompts[i % len(prompts)], performer, live=live)
    except Exception as e:  # noqa: BLE001 - a sound is optional, never fatal
        print(f"  ! sounds: {sound} not rendered ({str(e)[:80]})", file=sys.stderr)
        return None
    pcm = _shape(raw, max_s)
    heard = _whisper(pcm)
    path = _path(sound, performer, i)
    path.parent.mkdir(parents=True, exist_ok=True)
    if heard is not None and not heard_words_ok(heard):
        _rejected(sound, performer, i).write_text(heard, encoding="utf-8")
        print(f"  ! sounds: {sound}_{i} rejected (heard {heard!r})", file=sys.stderr)
        return None
    path.write_bytes(pcm)
    return pcm


def get(sound: str, voice: str | None = None, seed: str = "", *, build: bool = True,
        live: bool = False) -> bytes | None:
    """PCM for `sound` in a voice matching `voice`, or None. The variant is
    chosen from `seed` (the line's text) so a re-render picks the same one."""
    if sound not in BANK:
        return None
    performer = performer_for(voice)
    n = len(BANK[sound][0])
    start = hashlib.blake2b(seed.encode("utf-8"), digest_size=2).digest()[0] % n
    for k in range(n):
        i = (start + k) % n
        p = _path(sound, performer, i)
        if p.exists():
            return p.read_bytes()
    if not build or live:
        return None              # live calls never wait on a render; batch builds lazily
    for k in range(n):
        i = (start + k) % n
        if _rejected(sound, performer, i).exists():
            continue
        pcm = build_one(sound, performer, i)
        if pcm:
            return pcm
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="paralang.sounds", description="nonverbal sound bank")
    ap.add_argument("--build", action="store_true", help="render every sound/variant (Groq key)")
    ap.add_argument("--performer", choices=[FEMALE_PERFORMER, MALE_PERFORMER, "both"], default="both")
    ap.add_argument("--retry-rejected", action="store_true")
    a = ap.parse_args(argv)
    performers = [FEMALE_PERFORMER, MALE_PERFORMER] if a.performer == "both" else [a.performer]
    print(f"cache: {cache_dir()}")
    for performer in performers:
        for sound, (prompts, _) in BANK.items():
            for i in range(len(prompts)):
                if _path(sound, performer, i).exists():
                    status = "cached"
                elif _rejected(sound, performer, i).exists() and not a.retry_rejected:
                    status = "rejected earlier"
                elif a.build:
                    status = "ok" if build_one(sound, performer, i) else "FAILED/rejected"
                else:
                    status = "missing"
                print(f"{performer:7} {sound:8} {i}  {status}")


if __name__ == "__main__":
    main()
