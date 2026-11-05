# Fish-style cues on every engine: listen

Rendered 2026-11-05 with expressive mode on. The same two scripts go through each engine:

- `receptionist_*`: [warm] / [slightly sad] / [sighing] / [confident] / [hopeful] / [laughing]
- `story_*`: [mysterious] / [long-break] / [nervous] / [gasping] / [very excited]

| file | what happens to the cues |
|---|---|
| `*_orpheus.wav` | Orpheus acts every cue itself, verbatim |
| `*_aura.wav` | emotion via ending + gain; sigh / laugh / gasp spliced from the sound bank; long-break = 900 ms of silence |
| `*_edge.wav` | same, plus rate/pitch/volume knobs |

Whisper WER 0.05-0.10 on all six (the misses are "Riya" -> "Rhea" and "4" -> "four").

**Check by ear:**
- Does the spliced laugh/sigh sound like the same person as Andromeda/Aria? The bank is performed by Orpheus hannah, gender-matched but a different voice.
- With a spliced laugh, Aura still reads the author's "Ha," aloud after it. If that sounds wooden, drop the "Ha" from scripts aimed at Aura.
- Whisper heard a stray "Hi." at the end of `receptionist_aura`.

Sound bank: `~/.cache/paralang/sounds/{hannah,troy}/`. Rebuild with `python -m paralang.sounds --build --retry-rejected`.
