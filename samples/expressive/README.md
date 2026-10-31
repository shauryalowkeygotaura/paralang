# Expressive mode: listen before flipping the default

Rendered 2026-10-31 with `python -m paralang.bench`. Each pair is the same script, `_plain` (today's default) vs `_expressive` (`PARALANG_EXPRESSIVE=1`).

**Listen in this order:**
1. `hook_edge_plain.wav` vs `hook_edge_expressive.wav`: the biggest difference. Plain edge left 1.2-1.5 s of engine padding in every gap; expressive brings it to ~0.5 s (36.6 s -> 27.4 s, Whisper WER 0.044 -> 0).
2. `agent_aura_*` and `excited_aura_*`: tighter gaps, matched levels, room tone instead of dead air, and "!" endings on excited lines.
3. `*_orpheus_*`: Orpheus only gets the cleanup. Any difference you hear beyond that is Orpheus being random (it dropped "Guess what?" on one take).

## What expressive mode does
- **assemble.join**: trims each clip's engine padding, matches speech loudness across clips (gain capped at 9 dB), applies the emotion's volume as gain (Aura has no knob of its own), fills gaps with -62 dB room tone, and adds 8 ms fades.
- **perform.shape**: excited/urgent/cheerful lines end with "!", sad (and strongly apologetic) lines with "...". Words never change, and questions are never touched.

## What I tested and dropped
- **Aura paragraph flow** (one request per paragraph): 3 runs each. It NARROWED pitch range (hook 10.4 -> 7.3 semitones) with no clarity gain, so it stays an opt-in (`aura_pcm(..., flow=True)`), off even in expressive mode. Evidence: `experiments/aura-3x/`.
- On identical clips, assembly is pitch-neutral (9.31 vs 9.27 st). The pitch differences between Aura renders in the tables are Aura generating differently on each call, not this code.

## Reports
- `bench-all-engines.md`: first full run. Its Aura "expressive" rows used flow, which was dropped later.
- `bench-aura-final.md`: Aura with the final code.

Metrics are proxies: wer (clarity), f0_range_st / f0_move_st (melody), dyn_db (dynamics), pauses. Your ears decide.
