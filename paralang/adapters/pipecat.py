"""
Pipecat adapter: one FrameProcessor, placed right AFTER the STT service.

    Pipeline([transport.input(), stt, ParalangProcessor(), user_aggregator, llm, ...])

Why that spot (verified against pipecat 1.12):
  - STT passes InputAudioRawFrame downstream (audio_passthrough=True default),
    so we get the caller's raw audio.
  - The user aggregator broadcasts VADUserStarted/StoppedSpeakingFrame UPSTREAM
    too (that is how segmented STT hears VAD), so we see turn boundaries.
  - BotStarted/StoppedSpeakingFrame are pushed upstream by the output
    transport, so we can time the caller's response latency and barge-ins.
  - TranscriptionFrame passes through us before the aggregator, so we can
    prefix the [tone: ...] tag and the LLM reads it in context.

Timing: features are computed in a worker thread the moment VAD says the
caller stopped, i.e. in parallel with the STT request. By the time the
transcript comes back (~300 ms+) the features are usually ready, so the
added latency is ~0.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..analyzer import Analyzer, TurnResult
from ..lang import LanguageHint

PREROLL_S = 0.5          # VAD confirms speech only after start_secs; keep audio from before that
STALE_SEGMENT_S = 6.0    # a segment with no transcript by now was empty/dropped by STT
FEATURE_WAIT_S = 0.5     # never hold a transcript longer than this for features
MAX_TURN_S = 60          # cap buffered audio per turn (a stuck VAD must not eat RAM)


class ParalangProcessor(FrameProcessor):
    def __init__(self, analyzer: Analyzer | None = None, *, annotate: bool = True,
                 on_turn: Callable[[TurnResult], Awaitable[None] | None] | None = None,
                 on_language: Callable[[str | None], None] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = analyzer or Analyzer()
        self.annotate = annotate
        self.on_turn = on_turn
        # v0.3 #9: tell STT which language to expect (None = auto-detect)
        self.on_language = on_language
        self.language_hint = LanguageHint() if on_language else None
        self._buf = bytearray()
        self._preroll: deque[bytes] = deque()
        self._preroll_bytes = 0
        self._sr = 16000
        self._ch = 1
        self._speaking = False
        self._seg_meta: dict = {}
        self._bot_speaking = False
        self._bot_stopped_at: float | None = None
        self._pending: deque[tuple[float, asyncio.Task]] = deque()  # (stopped_at, features task)

    # ── frame routing ────────────────────────────────────────────────────────
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._on_audio(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._on_user_started(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._on_user_stopped()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_at = time.time()
        elif isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._on_transcript(frame)

        await self.push_frame(frame, direction)

    # ── audio + segmentation ─────────────────────────────────────────────────
    def _on_audio(self, frame: InputAudioRawFrame) -> None:
        self._sr, self._ch = frame.sample_rate, frame.num_channels
        if self._speaking:
            if len(self._buf) < MAX_TURN_S * self._sr * 2 * self._ch:
                self._buf.extend(frame.audio)
            return
        self._preroll.append(frame.audio)
        self._preroll_bytes += len(frame.audio)
        limit = int(PREROLL_S * self._sr) * 2 * self._ch
        while self._preroll and self._preroll_bytes - len(self._preroll[0]) >= limit:
            self._preroll_bytes -= len(self._preroll.popleft())

    def _on_user_started(self, frame: VADUserStartedSpeakingFrame) -> None:
        # Upstream and downstream copies of the same broadcast both reach us; ignore the second.
        if self._speaking:
            return
        self._speaking = True
        self._buf = bytearray(b"".join(self._preroll))
        self._preroll.clear()
        self._preroll_bytes = 0
        started_at = frame.timestamp - frame.start_secs
        latency = None
        if not self._bot_speaking and self._bot_stopped_at is not None:
            latency = max(0.0, started_at - self._bot_stopped_at)
            self._bot_stopped_at = None   # only the first segment after the bot answers it
        self._seg_meta = {"response_latency_s": latency, "barge_in": self._bot_speaking}

    def _on_user_stopped(self) -> None:
        if not self._speaking:
            return
        self._speaking = False
        audio, meta = bytes(self._buf), self._seg_meta
        self._buf = bytearray()
        sr, ch = self._sr, self._ch
        # Text is not known yet; word-based features are filled in at transcript time.
        task = asyncio.create_task(asyncio.to_thread(
            self.analyzer.features, audio, sr, "", num_channels=ch, **meta))
        self._drop_stale()   # also prune here: STT that returns nothing never triggers _on_transcript
        self._pending.append((time.time(), task))

    # ── transcript: join features + text, tag it ────────────────────────────
    def _drop_stale(self) -> None:
        now = time.time()
        while self._pending and now - self._pending[0][0] > STALE_SEGMENT_S:
            self._pending.popleft()[1].cancel()

    async def _on_transcript(self, frame: TranscriptionFrame) -> None:
        self._drop_stale()
        if not self._pending:
            return
        _, task = self._pending.popleft()
        t0 = time.perf_counter()
        try:
            feats = await asyncio.wait_for(task, FEATURE_WAIT_S)
        except Exception as e:  # noqa: BLE001 - never break the call for a tone hint
            logger.warning(f"paralang: features unavailable for this turn ({type(e).__name__})")
            return
        _apply_text(feats, frame.text)
        res = self.analyzer.score(feats, frame.text, (time.perf_counter() - t0) * 1000)
        if self.annotate and res.tag:
            frame.text = res.annotate(frame.text)
        logger.debug(f"paralang turn {res.turn}: {res.tag or '(no cues)'}")
        if self.language_hint is not None:
            change = self.language_hint.observe(res.text)   # the caller's words, untagged
            if change is not LanguageHint.UNCHANGED:
                logger.info(f"paralang: STT language hint -> {change or 'auto'}")
                self.on_language(change)
        if self.on_turn:
            out = self.on_turn(res)
            if asyncio.iscoroutine(out):
                await out


def _apply_text(feats, text: str) -> None:
    """Fill the word-based features once the transcript exists."""
    from ..features import _FILLER_RE, _WORD_RE, MIN_ACOUSTIC_S
    feats.words = len(_WORD_RE.findall(text or ""))
    feats.fillers = len(_FILLER_RE.findall(text or ""))
    feats.words_per_sec = (feats.words / feats.voiced_s
                           if feats.words and feats.voiced_s >= MIN_ACOUSTIC_S else None)


# ── STT that can genuinely auto-detect (v0.3 #9) ────────────────────────────
# Pipecat's GroqSTTService ALWAYS sends a language and defaults to "en", so a
# Hindi caller was transcribed as forced English (found 2026-10-22). This
# subclass sends no language until ParalangProcessor's LanguageHint sets one.
try:
    from pipecat.services.groq.stt import GroqSTTService as _GroqSTT
except ImportError:  # pragma: no cover - groq extra not installed
    _GroqSTT = None

if _GroqSTT is not None:
    class AutoLanguageGroqSTT(_GroqSTT):
        """Groq Whisper with a runtime language hint; None = Whisper auto-detects.

            stt = AutoLanguageGroqSTT(api_key=key, settings=AutoLanguageGroqSTT.Settings(
                model="whisper-large-v3-turbo"))
            ParalangProcessor(on_language=stt.set_language_hint)
        """

        def __init__(self, *args, language_hint: str | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.language_hint = language_hint

        def set_language_hint(self, lang: str | None) -> None:
            if lang not in (None, "hi", "en"):
                raise ValueError(f"unsupported hint {lang!r}")
            self.language_hint = lang

        async def _transcribe(self, audio: bytes):
            kwargs = {
                "file": ("audio.wav", audio, "audio/wav"),
                "model": self._settings.model,
                "response_format": "verbose_json" if self._include_prob_metrics else "json",
            }
            if self.language_hint:
                kwargs["language"] = self.language_hint
            if self._settings.prompt is not None:
                kwargs["prompt"] = self._settings.prompt
            if self._settings.temperature is not None:
                kwargs["temperature"] = self._settings.temperature
            return await self._client.audio.transcriptions.create(**kwargs)
