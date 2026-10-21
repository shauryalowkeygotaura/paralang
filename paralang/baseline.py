"""
baseline.py -- "how does THIS caller normally sound?"

The single most important design rule in paralang: compare a caller to
themselves, never to a fixed threshold. A soft-spoken aunty on a Jio line and
a loud guy on a headset differ by more than any emotion does. So each feature
becomes a z-score against the caller's own first few turns.

The baseline freezes after `warmup` turns on purpose: if it kept updating, a
caller who slowly gets more irritated would drag their own baseline along and
the drift would become invisible.
"""
from __future__ import annotations

import math

from .features import TurnFeatures

# Features that get z-scored, with a minimum std so two near-identical warmup
# turns don't turn a tiny change into a huge z. Units are each feature's own.
TRACKED = {
    "words_per_sec":     0.4,
    "pause_ratio":       0.05,
    "loudness":          1.5,    # dB (numpy RMS, same units in fast and full mode)
    "f0_median_st":      1.0,
    "f0_range_st":       1.0,
    "response_latency_s": 0.3,
    "noise_db":          2.0,    # dB; a jump means the caller moved somewhere louder
}


class Baseline:
    def __init__(self, warmup: int = 3):
        self.warmup = warmup
        self._vals: dict[str, list[float]] = {k: [] for k in TRACKED}
        self.turns_seen = 0

    @property
    def frozen(self) -> bool:
        return self.turns_seen >= self.warmup

    def update(self, f: TurnFeatures) -> None:
        if self.frozen:
            return
        self.turns_seen += 1
        for k in TRACKED:
            v = getattr(f, k)
            if v is not None:
                self._vals[k].append(float(v))

    def z(self, f: TurnFeatures) -> dict[str, float]:
        """z-scores for every tracked feature that has >=2 baseline samples.

        Missing keys mean "not enough history yet": cues must treat that as
        unknown, not as zero.
        """
        out = {}
        for k, floor in TRACKED.items():
            v, hist = getattr(f, k), self._vals[k]
            if v is None or len(hist) < 2:
                continue
            mean = sum(hist) / len(hist)
            std = math.sqrt(sum((h - mean) ** 2 for h in hist) / (len(hist) - 1))
            # floor is absolute; also floor at 10% of |mean| for scale-free features
            std = max(std, floor, 0.1 * abs(mean))
            out[k] = (float(v) - mean) / std
        return out
