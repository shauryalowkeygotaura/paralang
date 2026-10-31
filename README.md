# paralang

Voice agents read transcripts. Transcripts lose *how* something was said. paralang measures it, and shapes how the agent talks back.

**Listening.** Per caller turn it measures pauses, pace, pitch and loudness, and compares each caller to their own first three turns (no fixed thresholds, no emotion classifier). Out comes JSON plus a short `[tone: hesitant]` tag for the LLM. Cues: hesitant, uncertain_rise, long_think, rushed, flat, agitated, barged_in, noisy. Also: laughs and sighs, voicemail/IVR detection, backchannel vs real interruption, adaptive silence timeouts.

**Speaking.** Write `[warm] Hi there. [excited] Big news!` once and render it for any engine: Deepgram Aura-2, Groq Orpheus, edge-tts, or as a payload for SSML / Azure / ElevenLabs v3 / OpenAI / Gemini. Instructable engines act the emotion. Knob engines get rate/pitch/volume plus per-sentence variation and breathing gaps. `|` adds a beat and `*word*` adds emphasis.

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
