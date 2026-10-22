"""paralang -- hear HOW a caller speaks (pauses, pace, pitch, loudness), not just what.

Framework-agnostic core + thin adapters:
    from paralang import Analyzer                         # any agent
    from paralang.adapters.pipecat import ParalangProcessor  # live, Pipecat
    python -m paralang.adapters.offline call.wav --turns turns.json  # post-call
"""
from .analyzer import PHRASE, Analyzer, TurnResult
from .cues import Cue, derive_cues
from .delivery import SPEAKING_HINT, DeliveryPlanner, Prosody
from .emotion import EMOTIONS, Script, express
from .features import TurnFeatures, extract

# One line for any agent's system prompt so the LLM uses (and never speaks) the tags.
PROMPT_HINT = (
    "Caller turns may start with a [tone: ...] note measured from their voice "
    "(pauses, pace, pitch). Never read it aloud. Use it: if hesitant or unsure, "
    "slow down and offer one simple option; if rushed, be brief; if flat, ask one "
    "easy question to re-engage; if noisy line, use short sentences and offer to call back; if answering machine or IVR, "
    "leave your one-line voicemail message and end the call. A [coach: ...] note is an "
    "instruction for this one reply only: follow it, never read it aloud."
)

__all__ = ["Analyzer", "TurnResult", "TurnFeatures", "Cue", "extract", "derive_cues",
           "PHRASE", "PROMPT_HINT", "DeliveryPlanner", "Prosody", "SPEAKING_HINT",
           "EMOTIONS", "Script", "express"]
