"""
analyzer.py -- the one object every adapter talks to.

    an = Analyzer()
    res = an.analyze_turn(pcm, 16000, "haan... theek hai", response_latency_s=2.4)
    res.tag       -> "[tone: long pause before answering]"
    res.to_dict() -> the JSON contract (schema.json)

One Analyzer per call: it owns that caller's baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .baseline import Baseline
from .cues import Cue, derive_cues
from .features import TurnFeatures, extract

SCHEMA_VERSION = "1"

# Spoken-English phrasing of each cue for the LLM. Short: it is read every turn.
PHRASE = {
    "long_think":     "long pause before answering",
    "hesitant":       "hesitant",
    "rushed":         "speaking fast",
    "flat":           "flat, low energy",
    "agitated":       "agitated",
    "uncertain_rise": "sounds unsure",
    "barged_in":      "interrupted you",
}
IDLE_S = 0.6            # hybrid: background openSMILE starts this long after the last turn
IDLE_MAX_WAIT_S = 10.0  # ...or after this long regardless, so a busy stream still gets data
TAG_MIN_STRENGTH = 0.4   # weaker cues go in the JSON but not in the LLM's prompt


@dataclass
class TurnResult:
    turn: int
    text: str
    features: TurnFeatures
    z: dict[str, float]
    cues: list[Cue]
    baseline_ready: bool
    elapsed_ms: float
    ts: float = field(default_factory=time.time)

    @property
    def tag(self) -> str:
        strong = sorted((c for c in self.cues if c.strength >= TAG_MIN_STRENGTH),
                        key=lambda c: -c.strength)
        if not strong:
            return ""
        return "[tone: " + ", ".join(PHRASE.get(c.name, c.name) for c in strong) + "]"

    def annotate(self, text: str | None = None) -> str:
        """Transcript text with the tag prefixed (what the LLM should see)."""
        text = self.text if text is None else text
        return f"{self.tag} {text}".strip() if self.tag else text

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "turn": self.turn,
            "ts": round(self.ts, 3),
            "text": self.text,
            "tag": self.tag,
            "cues": [c.to_dict() for c in self.cues],
            "features": self.features.to_dict(),
            "z": {k: round(v, 3) for k, v in self.z.items()},
            "baseline_ready": self.baseline_ready,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


def _opensmile_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("opensmile") is not None


class _Shared:
    """One openSMILE worker and one "a turn is in flight" clock per PROCESS.

    Several concurrent calls in one process (a phone server) must not each
    spawn an openSMILE thread: they would load it N times and fight each other
    for CPU while another call is on its critical path. One serial worker, one
    shared idle signal: background work only runs when no call is measuring.
    """
    pool = None
    busy_until = 0.0
    lock = __import__("threading").Lock()

    @classmethod
    def get_pool(cls):
        with cls.lock:
            if cls.pool is None:
                from concurrent.futures import ThreadPoolExecutor
                from .features import prewarm
                cls.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paralang-smile")
                cls.pool.submit(prewarm)          # ~12 s cold load happens here, once
            return cls.pool

    @classmethod
    def mark_busy(cls, until: float) -> None:
        with cls.lock:
            cls.busy_until = max(cls.busy_until, until) if until > time.monotonic() else until


def _attach(feats: TurnFeatures, fut) -> None:
    """Merge background voice-quality results into the turn's features."""
    try:
        vq = fut.result()
    except Exception:  # noqa: BLE001 - enrichment is best-effort, logged in voice_quality
        return
    for k in ("jitter", "shimmer_db", "hnr_db"):
        if vq.get(k) is not None:
            setattr(feats, k, vq[k])
    if vq.get("f0_median_st") is not None:
        # kept beside the fast tracker's value so real calls can validate it
        feats.extras["smile_f0_median_st"] = round(vq["f0_median_st"], 3)


class Analyzer:
    def __init__(self, warmup_turns: int = 3, acoustic: bool | str = "hybrid"):
        """acoustic modes (nothing is dropped in the default):

        "hybrid" (default): numpy pitch + loudness on the critical path (~17 ms),
                  so cues and the tag are ready instantly; openSMILE then adds
                  jitter / shimmer / HNR / its own F0 in a background worker,
                  prewarmed at construction so turn 1 never pays the ~12 s load.
                  Falls back to "fast" if openSMILE is not installed.
        "full":   openSMILE on the critical path (blocking, ~150 ms warm).
        "fast":   numpy only.   False: timing/text features only.
        """
        self.baseline = Baseline(warmup=warmup_turns)
        mode = "hybrid" if acoustic is True else acoustic
        if mode == "hybrid" and not _opensmile_available():
            mode = "fast"
        self.acoustic = mode
        self.turns: list[TurnResult] = []
        self._pool = None
        self._enrich: list = []
        self._flush = False
        if mode in ("hybrid", "full"):
            self._pool = _Shared.get_pool()   # shared, prewarmed once per process

    def features(self, audio, sample_rate: int, text: str = "", **kw) -> TurnFeatures:
        """Measurement only (no state, safe in a worker thread). Lets live
        adapters run the slow part before the transcript even arrives.
        In hybrid mode this is the fast measurement; voice quality is queued."""
        if self.acoustic != "hybrid":
            return extract(audio, sample_rate, text, acoustic=self.acoustic, **kw)
        _Shared.mark_busy(time.monotonic() + 10)   # a turn is in flight: background waits
        try:
            feats = extract(audio, sample_rate, text, acoustic="fast", **kw)
        finally:
            _Shared.busy_until = time.monotonic() + IDLE_S
        if feats.f0_median_st is not None:   # enough voice to be worth measuring
            from .features import to_float_mono
            x = to_float_mono(audio, sample_rate, kw.get("num_channels", 1))
            fut = self._pool.submit(self._when_idle, x, sample_rate)
            fut.add_done_callback(lambda f, feats=feats: _attach(feats, f))
            self._enrich.append(fut)
        return feats

    def _when_idle(self, x, sr) -> dict:
        """Background voice-quality pass, started only once no turn has been
        measured for IDLE_S: it must never share the CPU with the fast path.
        Gives up waiting after IDLE_MAX_WAIT_S so a busy stream still gets data."""
        from .features import voice_quality
        deadline = time.monotonic() + IDLE_MAX_WAIT_S
        while time.monotonic() < min(_Shared.busy_until, deadline) and not self._flush:
            time.sleep(0.02)
        return voice_quality(x, sr)

    def wait_enrichment(self, timeout: float = 5.0) -> bool:
        """Block until queued voice-quality work is done (call at end of call,
        before exporting). Returns False if it timed out."""
        from concurrent.futures import wait
        pending = [f for f in self._enrich if not f.done()]
        if not pending:
            return True
        self._flush = True            # caller is waiting: stop deferring for idleness
        try:
            done, not_done = wait(pending, timeout=timeout)
        finally:
            self._flush = False
        return not not_done

    def export(self, timeout: float = 5.0) -> list[dict]:
        """Every turn as JSON, after waiting for background enrichment."""
        self.wait_enrichment(timeout)
        return [t.to_dict() for t in self.turns]

    def close(self) -> None:
        for f in self._enrich:
            f.cancel()                      # this call's queued work only; pool is shared

    def score(self, feats: TurnFeatures, text: str = "", elapsed_ms: float = 0.0) -> TurnResult:
        """Stateful part: z against baseline, derive cues, then grow the baseline."""
        z = self.baseline.z(feats)
        cues = derive_cues(feats, z)
        res = TurnResult(turn=len(self.turns) + 1, text=text, features=feats, z=z, cues=cues,
                         baseline_ready=self.baseline.frozen, elapsed_ms=elapsed_ms)
        self.baseline.update(feats)
        self.turns.append(res)
        return res

    def analyze_turn(self, audio, sample_rate: int, text: str = "", **kw) -> TurnResult:
        t0 = time.perf_counter()
        feats = self.features(audio, sample_rate, text, **kw)
        return self.score(feats, text, (time.perf_counter() - t0) * 1000)

    def call_signals(self, min_strength: float = 0.5, min_turns: int = 1) -> list[str]:
        """Distinct cue names seen across the call as scorer-friendly phrases
        ("tone: hesitant"). A cue counts once per call, however many turns."""
        counts: dict[str, int] = {}
        for t in self.turns:
            for c in t.cues:
                if c.strength >= min_strength:
                    counts[c.name] = counts.get(c.name, 0) + 1
        return [f"tone: {n}" for n, k in counts.items() if k >= min_turns]
