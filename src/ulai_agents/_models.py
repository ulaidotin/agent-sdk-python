"""Every value the server is known to accept, as str enums.

Validation rejects anything else before a call is made, unless
``allow_unlisted_values=True`` is passed to ``join`` or ``run``.

A value is listed only once the server is known to run it. Adding a model or
voice the server has started to support is one line in an enum here (and, for
a realtime or speech-to-text model, one in the table below it).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = [
    "VAD",
    "AgentState",
    "CascadeMode",
    "ClassifierBackend",
    "LLMModel",
    "Language",
    "Opening",
    "ParamType",
    "PipelineKind",
    "RealtimeModel",
    "Role",
    "STTModel",
    "Speaker",
    "TurnDetection",
    "UlaiLiveVoice",
    "Voice",
]


class _Value(str, Enum):
    """A str enum that prints as its value."""

    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, spec: str) -> str:
        return format(str(self.value), spec)


class RealtimeModel(_Value):
    """A speech-to-speech model: it hears the caller and answers in its own voice."""

    GEMINI_LIVE_2_5_FLASH_NATIVE_AUDIO = "gemini-live-2.5-flash-native-audio"
    """The server's default."""


class LLMModel(_Value):
    """The text model a Cascade thinks with."""

    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    """The cascade's default."""


class STTModel(_Value):
    """The speech-to-text model a Cascade listens with.

    Each model is served from particular regions; when ``Cascade.stt_region`` is
    empty the SDK picks the one the model was measured working in.
    """

    SHORT = "short"
    """The default: the fastest final transcript (about 0.6s), with interim results."""
    CHIRP_2 = "chirp_2"
    """Slightly more accurate, finals in about 1.5s, no interim results."""
    CHIRP_3 = "chirp_3"
    """Finals in about 1.6s, no interim results."""


class CascadeMode(_Value):
    """How a Cascade hears the caller."""

    STT = "stt"
    """Transcribe the caller and hand the text model the words. The default."""
    AUDIO = "audio"
    """Hand the caller's audio to the text model directly."""


class VAD(_Value):
    """The voice-activity detector a Cascade gates the caller's audio with."""

    SILERO = "silero"
    TEN = "ten"


class Voice(_Value):
    """A prebuilt voice, for Realtime and for a Cascade's text-to-speech."""

    ACHERNAR = "Achernar"
    ACHIRD = "Achird"
    ALGENIB = "Algenib"
    ALGIEBA = "Algieba"
    ALNILAM = "Alnilam"
    AOEDE = "Aoede"
    AUTONOE = "Autonoe"
    CALLIRRHOE = "Callirrhoe"
    CHARON = "Charon"
    DESPINA = "Despina"
    ENCELADUS = "Enceladus"
    ERINOME = "Erinome"
    FENRIR = "Fenrir"
    GACRUX = "Gacrux"
    IAPETUS = "Iapetus"
    KORE = "Kore"
    LAOMEDEIA = "Laomedeia"
    LEDA = "Leda"
    ORUS = "Orus"
    PUCK = "Puck"
    PULCHERRIMA = "Pulcherrima"
    RASALGETHI = "Rasalgethi"
    SADACHBIA = "Sadachbia"
    SADALTAGER = "Sadaltager"
    SCHEDAR = "Schedar"
    SULAFAT = "Sulafat"
    UMBRIEL = "Umbriel"
    VINDEMIATRIX = "Vindemiatrix"
    ZEPHYR = "Zephyr"
    ZUBENELGENUBI = "Zubenelgenubi"


class UlaiLiveVoice(_Value):
    """A voice from the Ulai Live API's catalogue."""

    ALBA = "alba"
    """The Ulai Live default."""


class Language(_Value):
    """The language a call is held in, as a BCP-47 code."""

    ENGLISH_INDIA = "en-IN"
    ENGLISH_US = "en-US"
    ENGLISH_UK = "en-GB"
    HINDI = "hi-IN"
    BENGALI = "bn-IN"
    GUJARATI = "gu-IN"
    KANNADA = "kn-IN"
    MALAYALAM = "ml-IN"
    MARATHI = "mr-IN"
    TAMIL = "ta-IN"
    TELUGU = "te-IN"


class ClassifierBackend(_Value):
    """What judged who answered a dialled call: a person, a voicemail, a ringtone.

    .. deprecated:: 0.2.1
       Ulai no longer detects voicemail, so ``Options.classifier`` has no
       effect on the call."""

    GEMINI_LIVE = "gemini-live"
    """The server's former default."""


class Opening(_Value):
    """Who speaks first."""

    AGENT = "agent"
    """The agent has the floor as soon as the call is up. The default."""
    CALLER = "caller"
    """The agent stays silent until the caller speaks."""


class Role(_Value):
    """Whose voice ``Session.inject_context`` speaks in."""

    USER = "user"
    """The text is something the model was told."""
    MODEL = "model"
    """The text is something the model itself said."""


class Transport(_Value):
    """How the agent server carries a room's audio to the media server. Your
    code never handles audio either way; this only picks the agent server's
    connection."""

    WEBSOCKET = "websocket"
    """Opus frames over a WebSocket. The default."""
    WEBRTC = "webrtc"
    """Opus over a WebRTC peer connection."""


class Speaker(_Value):
    """Who said something."""

    CALLER = "caller"
    AGENT = "agent"


class AgentState(_Value):
    """What the agent is doing."""

    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class PipelineKind(_Value):
    """The shape of the pipeline running a call, as the server reports it."""

    REALTIME = "realtime"
    SEMI_CASCADE = "semi-cascade"
    CASCADE = "cascade"


class TurnDetection(_Value):
    """Who decides when the caller has finished speaking."""

    SERVER = "server"
    """The model decides."""
    CLIENT = "client"
    """The engine decides, from voice activity."""


class ParamType(_Value):
    """The JSON type of a tool parameter."""

    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"


# The server's names for its backends. Chosen through the pipeline types,
# never set directly.
BACKEND_GEMINI_LIVE = "gemini-live"
BACKEND_GOOGLE_CASCADE = "google-cascade"
BACKEND_ULAI_LIVE = "ulai-live"

# Each realtime model and the server backend that runs it.
REALTIME_BACKENDS: dict[str, str] = {
    RealtimeModel.GEMINI_LIVE_2_5_FLASH_NATIVE_AUDIO.value: BACKEND_GEMINI_LIVE,
}

# Where each speech-to-text model is served when Cascade.stt_region is empty.
# Model, region and language are only valid in combination.
STT_REGIONS: dict[str, str] = {
    STTModel.SHORT.value: "global",
    STTModel.CHIRP_2.value: "asia-southeast1",
    STTModel.CHIRP_3.value: "eu",
}


def values(enum: type[Enum]) -> frozenset[str]:
    """The wire values of an enum."""
    return frozenset(str(m.value) for m in enum)


def wire(value: Any) -> str:
    """A listed value as the server reads it: None is empty, a member its value."""
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def check_listed(
    problems: list[str],
    label: str,
    value: Any,
    enum: type[Enum],
    lists: bool,
    hint: str,
) -> None:
    """Report value unless it is unset, a member of enum, or its exact value.

    lists=False skips the list check (allow_unlisted_values) but never the type
    check: no server can take a number where it reads a name.
    """
    if value is None:
        return
    if not isinstance(value, str):
        problems.append(f"{label} must be a string, not {type(value).__name__}")
        return
    v = wire(value)
    if lists and v and v not in values(enum):
        problems.append(f'unknown {label} "{v}": {hint}')
