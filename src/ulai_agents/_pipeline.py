"""Pipelines: how an agent hears and speaks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from ulai.agent.v1 import agent_pb2 as pb

from ._models import (
    BACKEND_GEMINI_LIVE,
    BACKEND_GOOGLE_CASCADE,
    BACKEND_ULAI_LIVE,
    REALTIME_BACKENDS,
    STT_REGIONS,
    VAD,
    CascadeMode,
    Language,
    LLMModel,
    RealtimeModel,
    STTModel,
    UlaiLiveVoice,
    Voice,
    check_listed,
    wire,
)

__all__ = ["Cascade", "Pipeline", "Realtime", "UlaiLive"]

_MS = timedelta(milliseconds=1)


@dataclass
class Realtime:
    """One speech-to-speech model hears the caller and answers in its own
    voice. The lowest latency, and the server's default."""

    model: RealtimeModel | str | None = None
    """The speech-to-speech model. Default: GEMINI_LIVE_2_5_FLASH_NATIVE_AUDIO."""

    def _model(self) -> str:
        return wire(self.model) or RealtimeModel.GEMINI_LIVE_2_5_FLASH_NATIVE_AUDIO.value

    def _problems(self, lists: bool) -> list[str]:
        if self.model is not None and not isinstance(self.model, str):
            return [f"Realtime.model must be a string, not {type(self.model).__name__}"]
        if lists and self._model() not in REALTIME_BACKENDS:
            return [f'unknown RealtimeModel "{self._model()}": use one of the agent.RealtimeModel members']
        return []

    def _apply(self, p: pb.AgentProfile) -> None:
        # An unlisted model (allow_unlisted_values) runs on the realtime backend.
        p.ai_backend = REALTIME_BACKENDS.get(self._model(), BACKEND_GEMINI_LIVE)
        p.model = self._model()


@dataclass(kw_only=True)
class Cascade:
    """Three stages: speech-to-text, a text model, then text-to-speech. More
    control over each stage than Realtime, for some extra latency per turn.

    Every field is optional. None (or zero) is the server's default.
    """

    mode: CascadeMode | str | None = None
    """How the cascade hears the caller. Default: CascadeMode.STT."""

    llm: LLMModel | str | None = None
    """The text model. Default: LLMModel.GEMINI_2_5_FLASH."""
    llm_region: str = ""
    llm_thinking: bool | None = None
    """Let the text model think before answering. Slower."""
    llm_temperature: float | None = None
    """The text model's temperature when Agent.temperature is not set."""

    stt: STTModel | str | None = None
    """The speech-to-text model. Default: STTModel.SHORT."""
    stt_region: str = ""
    """Default: the region the model is served from."""
    stt_languages: Sequence[Language | str] = ()
    """The languages to listen for. Several turn on language detection.
    Default: Agent.language."""
    keywords: Sequence[str] = ()
    """Words speech-to-text should favour: names, products."""

    tts_voice: Voice | str | None = None
    """The speaking voice. Default: Agent.voice."""
    tts_region: str = ""
    tts_language: Language | str | None = None
    """The speaking language when Agent.language is not set."""
    tts_sample_rate: int | None = None
    """The speech sample rate in Hz. Default: 24000."""

    vad: VAD | str | None = None
    """The voice-activity detector the caller's audio is gated with."""
    vad_confidence: float | None = None
    """0 to 1."""
    vad_min_volume: float | None = None
    """0 to 1."""

    turn_model: bool | None = None
    """Use a model, not just silence, to decide when the caller has finished."""
    turn_model_max_silence: timedelta | None = None
    """End the caller's turn after this much silence even when the turn model
    is unsure."""

    transcript_wait: timedelta | None = None
    """How long to wait for the final transcript."""
    endpoint_silence: timedelta | None = None
    """Silence that ends an utterance."""
    min_utterance: timedelta | None = None
    """Shorter speech is ignored."""
    max_utterance: timedelta | None = None
    """Longer speech is cut into turns."""
    prefix_pad: timedelta | None = None
    """Audio kept from before speech was detected."""

    max_history_turns: int | None = None
    """How many turns of the conversation the text model sees."""

    _DURATIONS = (
        "turn_model_max_silence",
        "transcript_wait",
        "endpoint_silence",
        "min_utterance",
        "max_utterance",
        "prefix_pad",
    )

    def _problems(self, lists: bool) -> list[str]:
        out: list[str] = []
        p = "Cascade."
        check_range(out, p + "llm_temperature", self.llm_temperature, 0, 2)
        check_listed(out, p + "mode", self.mode, CascadeMode, lists,
                     "use agent.CascadeMode.STT or agent.CascadeMode.AUDIO")
        check_listed(out, p + "llm", self.llm, LLMModel, lists, "use one of the agent.LLMModel members")
        check_listed(out, p + "stt", self.stt, STTModel, lists, "use one of the agent.STTModel members")
        if isinstance(self.stt_languages, str) or not isinstance(self.stt_languages, Sequence):
            out.append(f"{p}stt_languages must be a list of languages")
        else:
            for lang in self.stt_languages:
                check_listed(out, p + "stt_languages entry", lang, Language, lists,
                             "use the agent.Language members")
        if isinstance(self.keywords, str) or not isinstance(self.keywords, Sequence):
            out.append(f"{p}keywords must be a list of strings")
        check_listed(out, p + "tts_voice", self.tts_voice, Voice, lists, "use one of the agent.Voice members")
        check_listed(out, p + "tts_language", self.tts_language, Language, lists,
                     "use one of the agent.Language members")
        check_listed(out, p + "vad", self.vad, VAD, lists, "use agent.VAD.SILERO or agent.VAD.TEN")
        for text in ("llm_region", "stt_region", "tts_region"):
            if not isinstance(getattr(self, text), str):
                out.append(f"{p}{text} must be a string")
        # The numbers no server could accept out of range, whatever its lists.
        check_range(out, p + "vad_confidence", self.vad_confidence, 0, 1)
        check_range(out, p + "vad_min_volume", self.vad_min_volume, 0, 1)
        for name in ("tts_sample_rate", "max_history_turns"):
            v = getattr(self, name)
            if v is None:
                continue
            if not isinstance(v, int) or isinstance(v, bool):
                out.append(f"{p}{name} must be an int, not {type(v).__name__}")
            elif v < 0:
                out.append(f"{p}{name} cannot be negative")
        for name in self._DURATIONS:
            check_duration(out, p + name, getattr(self, name), negative_ok=False)
        return out

    def _apply(self, p: pb.AgentProfile) -> None:
        p.ai_backend = BACKEND_GOOGLE_CASCADE
        cc = pb.CascadeConfig(
            mode=wire(self.mode),
            llm_model=wire(self.llm),
            llm_region=self.llm_region,
            stt_model=wire(self.stt),
            stt_region=self.stt_region,
            stt_languages=[wire(lang) for lang in self.stt_languages],
            stt_keywords=list(self.keywords),
            tts_voice=wire(self.tts_voice),
            tts_region=self.tts_region,
            tts_language=wire(self.tts_language),
            vad=wire(self.vad),
        )
        _opt(cc, "llm_thinking", self.llm_thinking)
        _opt(cc, "llm_temperature", self.llm_temperature)
        _opt(cc, "tts_rate", self.tts_sample_rate or None)
        _opt(cc, "vad_confidence", self.vad_confidence)
        _opt(cc, "vad_min_volume", self.vad_min_volume)
        _opt(cc, "turn_model", self.turn_model)
        _opt(cc, "turn_model_max_silence_ms", opt_ms(self.turn_model_max_silence))
        _opt(cc, "transcript_wait_ms", opt_ms(self.transcript_wait))
        _opt(cc, "endpoint_silence_ms", opt_ms(self.endpoint_silence))
        _opt(cc, "min_utterance_ms", opt_ms(self.min_utterance))
        _opt(cc, "max_utterance_ms", opt_ms(self.max_utterance))
        _opt(cc, "prefix_pad_ms", opt_ms(self.prefix_pad))
        _opt(cc, "max_history_turns", self.max_history_turns or None)
        # A model only works in some regions, so naming one without a region
        # would fail on the call. Fill in the region it is served from.
        if not cc.stt_region and cc.stt_model:
            cc.stt_region = STT_REGIONS.get(cc.stt_model, "")
        p.cascade.CopyFrom(cc)


@dataclass(kw_only=True)
class UlaiLive:
    """The call runs on the Ulai Live API."""

    url: str = ""
    """The API's host, e.g. "live.example.com". Empty is the server's default."""
    insecure: bool = False
    """Dial without TLS. Local testing only."""
    voice: UlaiLiveVoice | str | None = None
    """The speaking voice. Default: UlaiLiveVoice.ALBA."""
    output_rate: int | None = None
    """The sample rate the API sends audio at, in Hz."""

    def _problems(self, lists: bool) -> list[str]:
        out: list[str] = []
        check_listed(out, "UlaiLive.voice", self.voice, UlaiLiveVoice, lists,
                     "use one of the agent.UlaiLiveVoice members")
        if self.output_rate is not None:
            if not isinstance(self.output_rate, int) or isinstance(self.output_rate, bool):
                out.append(f"UlaiLive.output_rate must be an int, not {type(self.output_rate).__name__}")
            elif self.output_rate < 0:
                out.append("UlaiLive.output_rate cannot be negative")
        if not isinstance(self.url, str):
            out.append("UlaiLive.url must be a string")
        return out

    def _apply(self, p: pb.AgentProfile) -> None:
        p.ai_backend = BACKEND_ULAI_LIVE
        ss = pb.SpeechServiceConfig(url=self.url, insecure=self.insecure, voice=wire(self.voice))
        _opt(ss, "output_rate", self.output_rate or None)
        p.speech_service.CopyFrom(ss)


Pipeline = Realtime | Cascade | UlaiLive
"""How an agent hears and speaks."""

PIPELINES = (Realtime, Cascade, UlaiLive)


def _opt(msg: Any, name: str, value: Any) -> None:
    """Set an optional field, leaving it unset (the server's default) for None."""
    if value is not None:
        setattr(msg, name, value)


def opt_ms(d: timedelta | None) -> int | None:
    """d in milliseconds, at least 1, or None for zero so the server keeps its default."""
    if not d:
        return None
    return max(d // _MS, 1)


def check_range(out: list[str], label: str, v: Any, lo: float, hi: float) -> None:
    if v is None:
        return
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        out.append(f"{label} must be a number, not {type(v).__name__}")
    elif not lo <= v <= hi:
        out.append(f"{label} {v} is outside {lo} to {hi}")


def check_duration(out: list[str], label: str, d: Any, *, negative_ok: bool) -> None:
    if d is None:
        return
    if not isinstance(d, timedelta):
        out.append(f"{label} must be a datetime.timedelta, not {type(d).__name__}")
    elif d < timedelta(0) and not negative_ok:
        out.append(f"{label} cannot be negative")

