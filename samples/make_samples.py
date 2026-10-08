"""Render one agent reply three ways so you can HEAR the delivery layer.

    ../jio-outbound/local_agent/.venv-rt/Scripts/python samples/make_samples.py
-> samples/1_flat.wav, 2_human_neutral.wav, 3_human_to_hesitant_caller.wav, 4_human_to_rushed_caller.wav
"""
import asyncio
import subprocess
import sys
from pathlib import Path

import edge_tts

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paralang import Cue, DeliveryPlanner  # noqa: E402

VOICE = "en-IN-NeerjaNeural"
SR = 24000
OUT = Path(__file__).parent
REPLY = ["Achha... samajh gayi.", "Aapka plan kal expire ho raha hai, isliye call kiya.",
         "Kya main aapko ek chhota sa offer WhatsApp pe bhej doon?", "Koi jaldi nahi hai."]


async def synth(text: str, **kw) -> bytes:
    mp3 = bytearray()
    async for c in edge_tts.Communicate(text, VOICE, **kw).stream():
        if c.get("type") == "audio":
            mp3.extend(c["data"])
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1",
                        "-ar", str(SR), "pipe:1"], input=bytes(mp3), capture_output=True, check=True)
    return r.stdout


def write_wav(path: Path, pcm: bytes) -> None:
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm)


async def render(name: str, cues, human: bool) -> None:
    if not human:
        pcm = await synth(" ".join(REPLY), rate="+5%")
    else:
        planner, pcm = DeliveryPlanner(base_rate_pct=5), b""
        for s in REPLY:
            p = planner.plan(s, cues, reply_id="demo")
            print(f"  {name}: {p}  {s!r}")
            pcm += b"\x00\x00" * (SR * p.pause_before_ms // 1000) + await synth(s, **p.edge())
    write_wav(OUT / f"{name}.wav", pcm)


async def main():
    await render("1_flat", None, human=False)
    await render("2_human_neutral", [], human=True)
    await render("3_human_to_hesitant_caller", [Cue("hesitant", 1.0)], human=True)
    await render("4_human_to_rushed_caller", [Cue("rushed", 1.0)], human=True)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.wav"))))


if __name__ == "__main__":
    asyncio.run(main())
