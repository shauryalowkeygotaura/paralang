# paralang

Voice agents read transcripts. Transcripts lose *how* something was said. paralang measures it, and shapes how the agent talks back.

**Listening.** Per caller turn it measures pauses, pace, pitch and loudness, and compares each caller to their own first three turns (no fixed thresholds, no emotion classifier). Out comes JSON plus a short `[tone: hesitant]` tag for the LLM. Cues: hesitant, uncertain_rise, long_think, rushed, flat, agitated, barged_in, noisy. Also: laughs and sighs, voicemail/IVR detection, backchannel vs real interruption, adaptive silence timeouts.

**Speaking.** Write `[warm] Hi there. [excited] Big news!` once and render it for any engine: Deepgram Aura-2, Groq Orpheus, edge-tts, or as a payload for SSML / Azure / ElevenLabs v3 / OpenAI / Gemini. Instructable engines act the emotion. Knob engines get rate/pitch/volume plus per-sentence variation and breathing gaps. `|` adds a beat and `*word*` adds emphasis.

## Cues (Fish Audio style, on every engine)

```
[warm] Hi, this is Riya. [slightly sad] Oh no, a broken tooth? [sighing] That sounds painful.
[angry][shouting] Get out! [soft tone] Okay. This is [emphasis] really important. Wait [break] what?
```

- Emotions: 34 named emotions plus about 150 synonyms (furious, wistful, deadpan...), with intensity words (slightly, very, extremely) or an explicit `[excited:0.5]`.
- Tones: whispering, soft tone, shouting, slowly, in a hurry tone. Tones stack with an emotion.
- Sounds: laughing, chuckling, sighing, gasping, breath, yawning, sobbing, groaning, clear throat, hmm.
- Pacing: `[break]`, `[long-break]`, `[emphasis] word`, `|`, `*word*`.

Acting engines (Orpheus, Fish, ElevenLabs) receive the cues verbatim. Aura and edge receive them as prosody plus **real nonverbal clips** from a cached, Whisper-gated sound bank (`python -m paralang.sounds --build`). Brackets are never read aloud. `python -m paralang.markup` can add cues to plain text.

## Install

```bash
pip install -e ".[acoustic,say]"      # + ".[pipecat]" for the live processor
```

## Use

```python
from paralang import Analyzer, PROMPT_HINT   # PROMPT_HINT goes in the system prompt
an = Analyzer()                               # one per call
turn = an.analyze_turn(pcm, sample_rate=16000, text="umm... I think so?")
llm_input = turn.annotate()                   # "[tone: hesitant] umm... I think so?" once the
                                              # caller's 3-turn baseline exists
```

```bash
python -m paralang.say "[warm] Namaste ji. [excited] Aaj bada offer hai!" -o hi.wav
python -m paralang.say "..." --engine orpheus -v troy --expressive
python -m paralang.adapters.offline call.wav --turns turns.json     # post-call analysis
python -m paralang.bench                                            # engine comparison
```

Pipecat: put `paralang.adapters.pipecat.ParalangProcessor` right after STT.

Keys come from the environment: `DEEPGRAM_API_KEY` (Aura), `GROQ_API_KEY` (Orpheus, Whisper). edge-tts needs no key, but needs ffmpeg.

## Tests

```bash
python -m pytest -q
```

A few tests skip unless sibling demo repos are checked out.

MIT licensed.
