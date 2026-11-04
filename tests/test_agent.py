"""The agent: validation, the profile it becomes on the wire, and secrets."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from fakeserver import ROOM, check_balance, support
from ulai.agent.v1 import agent_pb2 as pb

import ulai_agents as agent


def problems(a: agent.Agent, **kw) -> str:
    with pytest.raises(agent.ValidationError) as e:
        a.validate(**kw)
    return str(e.value)


# ── 2. Agent.validate ─────────────────────────────────────────────────────────


def test_a_full_agent_is_valid():
    support().validate()


@pytest.mark.parametrize(
    "change, want",
    [
        ({"id": ""}, "Agent.id is required"),
        ({"instructions": ""}, "Agent.instructions is required"),
        ({"cloud": None}, "Agent.cloud needs a project and credentials_json"),
        ({"cloud": agent.CloudAccount(project="p")}, "Agent.cloud needs"),
        ({"cloud": agent.CloudAccount(project="", credentials_json="{}")}, "Agent.cloud needs"),
        ({"cloud": None, "pipeline": agent.Cascade()}, "Agent.cloud needs"),
        ({"cloud": None, "pipeline": None}, "Agent.cloud needs"),
        ({"cloud": "key.json"}, "Agent.cloud must be an agent.CloudAccount"),
        ({"max_duration": timedelta(milliseconds=500)}, "at least a second"),
        ({"max_duration": timedelta(seconds=-1)}, "max_duration cannot be negative"),
        ({"max_duration": 600}, "must be a datetime.timedelta"),
        ({"voice": "Pukc"}, 'unknown Agent.voice "Pukc"'),
        ({"voice": agent.Language.HINDI}, 'unknown Agent.voice "hi-IN"'),
        ({"voice": 7}, "Agent.voice must be a string"),
        ({"language": "xx-XX"}, 'unknown Agent.language "xx-XX"'),
        ({"pipeline": agent.Realtime(model="gpt-x")}, 'unknown RealtimeModel "gpt-x"'),
        ({"pipeline": agent.Cascade(llm="gpt-x")}, 'unknown Cascade.llm "gpt-x"'),
        ({"pipeline": agent.Cascade(stt="long")}, 'unknown Cascade.stt "long"'),
        ({"pipeline": agent.Cascade(vad="webrtc")}, 'unknown Cascade.vad "webrtc"'),
        ({"pipeline": agent.Cascade(mode="text")}, 'unknown Cascade.mode "text"'),
        ({"pipeline": agent.Cascade(stt_languages=["xx"])}, 'unknown Cascade.stt_languages entry "xx"'),
        ({"pipeline": agent.Cascade(tts_voice="Bob")}, 'unknown Cascade.tts_voice "Bob"'),
        ({"pipeline": agent.Cascade(tts_language="xx")}, 'unknown Cascade.tts_language "xx"'),
        ({"pipeline": agent.UlaiLive(voice="bob")}, 'unknown UlaiLive.voice "bob"'),
        ({"pipeline": "realtime"}, "use agent.Realtime, agent.Cascade or agent.UlaiLive"),
        ({"temperature": 3.0}, "Agent.temperature 3.0 is outside 0 to 2"),
        ({"temperature": -0.1}, "outside 0 to 2"),
        ({"opening": "whoever"}, 'unknown Agent.opening "whoever"'),
        ({"tools": [agent.tool_from_schema("check balance", "d")]}, "tool name 'check balance' must start"),
        ({"tools": [agent.tool_from_schema("t\n", "d")]}, "tool name 't\\n' must start"),
        ({"tools": [agent.tool_from_schema("x" * 65, "d")]}, "at most 64 characters"),
        ({"tools": [agent.tool_from_schema("t", " ")]}, 'tool "t" needs a description'),
        ({"tools": [check_balance, check_balance]}, 'tool "check_balance" is declared twice'),
        ({"tools": [lambda: None]}, "not a tool: declare it with @agent.tool"),
        ({"options": agent.Options(classifier="x")}, 'unknown Options.classifier "x"'),
        (
            {"options": agent.Options(first_words_timeout=timedelta(seconds=-1))},
            "first_words_timeout cannot be negative",
        ),
    ],
)
def test_validate_rejects(change, want):
    a = support(**change)
    assert want in problems(a)


def test_validate_lists_every_problem():
    a = agent.Agent(voice="Pukc", temperature=9)
    with pytest.raises(agent.ValidationError) as e:
        a.validate()
    err = e.value
    assert len(err.problems) == 5, err.problems  # id, instructions, cloud, voice, temperature
    assert str(err).startswith("5 problems:")
    assert isinstance(err, ValueError) and isinstance(err, agent.AgentError)


def test_tool_argument_types_are_checked():
    @agent.tool(description="d")
    def nested(address: dict) -> None: ...

    assert "use str, bool, int, float or a Literal of strings" in problems(support(tools=[nested]))


def test_the_smallest_valid_agent():
    @agent.tool(description="Ping.")
    def ping() -> None: ...

    agent.Agent(
        id="a",
        instructions="Be brief.",
        cloud=agent.CloudAccount(project="p", credentials_json="{}"),
        tools=[ping],
    ).validate()


def test_ulai_live_needs_cloud_only_for_voicemail_detection():
    a = agent.Agent(id="a", instructions="i", pipeline=agent.UlaiLive())
    a.validate()  # no Google model, so no cloud
    a.options = agent.Options(voicemail_detection=False)
    a.validate()
    a.options = agent.Options(voicemail_detection=True)
    assert "Agent.cloud needs" in problems(a)


def test_listed_values_accept_a_member_or_its_exact_value():
    support(voice="Kore", language="hi-IN", opening="caller").validate()
    support(voice=agent.Voice.PUCK, language=agent.Language.TAMIL, opening=agent.Opening.AGENT).validate()
    assert "unknown Agent.voice" in problems(support(voice="kore"))  # case matters
    assert "unknown Agent.voice" in problems(support(voice="KORE"))  # the member name is not the value


def test_cascade_ranges():
    for cascade, want in [
        (agent.Cascade(vad_confidence=1.5), "Cascade.vad_confidence 1.5 is outside 0 to 1"),
        (agent.Cascade(vad_min_volume=-1), "Cascade.vad_min_volume -1 is outside 0 to 1"),
        (agent.Cascade(llm_temperature=3), "Cascade.llm_temperature 3 is outside 0 to 2"),
        (agent.Cascade(tts_sample_rate=-1), "Cascade.tts_sample_rate cannot be negative"),
        (agent.Cascade(max_history_turns=-1), "Cascade.max_history_turns cannot be negative"),
        (agent.Cascade(prefix_pad=timedelta(milliseconds=-1)), "Cascade.prefix_pad cannot be negative"),
        (agent.Cascade(endpoint_silence=0.5), "Cascade.endpoint_silence must be a datetime.timedelta"),
        (agent.UlaiLive(output_rate=-1), "UlaiLive.output_rate cannot be negative"),
    ]:
        a = support(pipeline=cascade)
        assert want in problems(a)
        # Ranges and structure are checked even for unlisted values.
        assert want in problems(a, allow_unlisted_values=True)


# ── 3. Pipelines on the wire ──────────────────────────────────────────────────


def test_realtime_profile():
    p = support()._profile()
    assert p.ai_backend == "gemini-live"
    assert p.model == "gemini-live-2.5-flash-native-audio"
    assert not p.HasField("cascade") and not p.HasField("speech_service")


def test_realtime_defaults_its_model():
    p = support(pipeline=agent.Realtime())._profile()
    assert (p.ai_backend, p.model) == ("gemini-live", "gemini-live-2.5-flash-native-audio")


def test_no_pipeline_is_the_servers_default():
    p = support(pipeline=None)._profile()
    assert (p.ai_backend, p.model) == ("", "")


def test_cascade_fills_the_stt_region():
    a = support(
        pipeline=agent.Cascade(
            mode=agent.CascadeMode.AUDIO,
            llm=agent.LLMModel.GEMINI_2_5_FLASH,
            stt=agent.STTModel.CHIRP_2,
            vad=agent.VAD.TEN,
            stt_languages=[agent.Language.HINDI, agent.Language.ENGLISH_INDIA],
            keywords=["policy"],
            tts_voice=agent.Voice.PUCK,
            endpoint_silence=timedelta(milliseconds=500),
            turn_model=True,
            max_history_turns=8,
        )
    )
    a.validate()
    p = a._profile()
    c = p.cascade
    assert p.ai_backend == "google-cascade" and p.model == ""
    assert (c.mode, c.llm_model, c.stt_model, c.stt_region, c.vad, c.tts_voice) == (
        "audio", "gemini-2.5-flash", "chirp_2", "asia-southeast1", "ten", "Puck",
    )
    assert list(c.stt_languages) == ["hi-IN", "en-IN"] and list(c.stt_keywords) == ["policy"]
    assert c.endpoint_silence_ms == 500 and c.turn_model and c.max_history_turns == 8
    # Durations and numbers left unset stay unset, so the server keeps its defaults.
    for name in ("transcript_wait_ms", "prefix_pad_ms", "tts_rate", "llm_temperature", "llm_thinking",
                 "vad_confidence", "turn_model_max_silence_ms"):
        assert not c.HasField(name), name


@pytest.mark.parametrize(
    "stt, region, want",
    [
        (agent.STTModel.SHORT, "", "global"),
        (agent.STTModel.CHIRP_2, "", "asia-southeast1"),
        (agent.STTModel.CHIRP_3, "", "eu"),
        (agent.STTModel.CHIRP_3, "us", "us"),  # an explicit region wins
        (None, "", ""),  # no model, no region
    ],
)
def test_cascade_stt_region(stt, region, want):
    p = support(pipeline=agent.Cascade(stt=stt, stt_region=region))._profile()
    assert p.cascade.stt_region == want


def test_cascade_zero_is_unset():
    p = support(pipeline=agent.Cascade(
        tts_sample_rate=0, max_history_turns=0, transcript_wait=timedelta(0), turn_model_max_silence=None,
    ))._profile()
    for name in ("tts_rate", "max_history_turns", "transcript_wait_ms", "turn_model_max_silence_ms"):
        assert not p.cascade.HasField(name), name


def test_cascade_durations_round_down_to_at_least_a_millisecond():
    p = support(pipeline=agent.Cascade(
        prefix_pad=timedelta(microseconds=300), min_utterance=timedelta(milliseconds=250, microseconds=900),
    ))._profile()
    assert p.cascade.prefix_pad_ms == 1
    assert p.cascade.min_utterance_ms == 250


def test_cascade_carries_its_own_temperature_and_language():
    a = support(pipeline=agent.Cascade(llm_temperature=0.4, tts_language=agent.Language.TAMIL, llm_thinking=False))
    a.validate()
    c = a._profile().cascade
    assert c.llm_temperature == pytest.approx(0.4) and c.tts_language == "ta-IN"
    assert c.HasField("llm_thinking") and c.llm_thinking is False


def test_ulai_live_pipeline():
    p = support(pipeline=agent.UlaiLive(
        url="live.example.com", voice=agent.UlaiLiveVoice.ALBA, output_rate=16000, insecure=True,
    ))._profile()
    assert p.ai_backend == "ulai-live" and p.model == ""
    s = p.speech_service
    assert (s.url, s.voice, s.output_rate, s.insecure) == ("live.example.com", "alba", 16000, True)
    assert not agent.UlaiLive()._problems(True)
    p = support(pipeline=agent.UlaiLive())._profile()
    assert not p.speech_service.HasField("output_rate")


def test_max_duration_and_first_words_timeout_on_the_wire():
    p = support(max_duration=timedelta(seconds=90, milliseconds=900))._profile()
    assert p.max_call_duration_seconds == 90
    a = support(max_duration=None)
    a.options = agent.Options(first_words_timeout=timedelta(milliseconds=2500))
    p = a._profile()
    assert p.max_call_duration_seconds == 0 and p.initial_engagement_deadline_ms == 2500


def test_optional_switches_are_sent_only_when_set():
    opts = agent.Options(
        interruptible=True, greeting_interruptible=False, backchannel_gate=False, silence_monitor=True,
        end_call_tool=False, voicemail_detection=False, replay_lost_utterances=True, minimal_audio_processing=False,
    )
    p = support(options=opts)._profile()
    for name, want in [
        ("allow_interruptions", True), ("greeting_interruptible", False), ("backchannel_gate", False),
        ("silence_monitor", True), ("end_call_enabled", False), ("amd_enabled", False),
        ("lost_utterance_replay", True), ("minimal_audio_processing", False),
    ]:
        assert p.HasField(name) and getattr(p, name) is want, name
    p = support(options=agent.Options(), temperature=None)._profile()
    for name in ("allow_interruptions", "greeting_interruptible", "backchannel_gate", "silence_monitor",
                 "end_call_enabled", "amd_enabled", "lost_utterance_replay", "minimal_audio_processing",
                 "temperature"):
        assert not p.HasField(name), name


async def test_join_sends_the_whole_agent(fake):
    """The full join request, field by field, against one built by hand."""
    a = support(closing="Bye", persona="Warm.")
    a.options = agent.Options(
        interruptible=False, greeting_interruptible=True, first_words_timeout=timedelta(seconds=4),
        backchannel_gate=True, silence_monitor=False, end_call_tool=True, voicemail_detection=True,
        replay_lost_utterances=False, background_noise=True, live_transcription=True, context_compression=True,
        minimal_audio_processing=True, classifier=agent.ClassifierBackend.GEMINI_LIVE, workspace_id="ws-1",
    )
    seat = await fake.client.join(ROOM, a, participant_id="agent-p", fields={"call_id": "c-9", "n": 2})
    assert (seat.room, seat.agent_id, seat.participant_id, seat.connection_id, seat.node) == (
        "room-1", "support", "p-1", "c-1", "node:443",
    )

    want = pb.JoinRoomRequest(
        agent_id="support",
        room_id="room-1",
        participant_id="agent-p",
        custom_fields_json='{"call_id":"c-9","n":2}',
        opening=pb.OPENING_TURN_CALLER,
        room_url="https://cp.test",
        room_token="tok",
        profile=pb.AgentProfile(
            prompt="You are a support agent.",
            greeting="Hello",
            closing="Bye",
            persona_text="Warm.",
            voice="Kore",
            model="gemini-live-2.5-flash-native-audio",
            language="hi-IN",
            temperature=0.3,
            tools=[pb.ToolSpec(
                name="check_balance",
                description="Look up a balance.",
                parameters={
                    "account": pb.ToolParam(type="string", description="The account number."),
                    "kind": pb.ToolParam(type="string", description="Which balance.", enum=["savings", "current"]),
                    "limit": pb.ToolParam(type="integer"),
                },
                required=["account"],
            )],
            allow_interruptions=False,
            greeting_interruptible=True,
            max_call_duration_seconds=600,
            opening=pb.OPENING_TURN_CALLER,
            backchannel_gate=True,
            silence_monitor=False,
            initial_engagement_deadline_ms=4000,
            background_noise=True,
            workspace_id="ws-1",
            live_transcription=True,
            ai_backend="gemini-live",
            ai_classify_backend="gemini-live",
            provider_project="proj",
            provider_region="us-central1",
            provider_credentials="{}",
            amd_enabled=True,
            minimal_audio_processing=True,
            lost_utterance_replay=False,
            end_call_enabled=True,
            context_compression=True,
        ),
    )
    assert len(fake.joins) == 1
    assert fake.joins[0] == want, f"\n got: {fake.joins[0]}\nwant: {want}"


async def test_join_sends_a_cascade(fake):
    a = support(opening=agent.Opening.AGENT, pipeline=agent.Cascade(
        llm=agent.LLMModel.GEMINI_2_5_FLASH, llm_region="europe-west4", llm_thinking=True, llm_temperature=0.5,
        stt=agent.STTModel.SHORT, stt_languages=[agent.Language.TAMIL], keywords=["Ulai"],
        tts_voice=agent.Voice.CHARON, tts_region="asia-south1", tts_language=agent.Language.TAMIL,
        tts_sample_rate=16000, vad=agent.VAD.SILERO, vad_confidence=0.6, vad_min_volume=0.2, turn_model=False,
        turn_model_max_silence=timedelta(seconds=2), transcript_wait=timedelta(milliseconds=300),
        endpoint_silence=timedelta(milliseconds=600), min_utterance=timedelta(milliseconds=200),
        max_utterance=timedelta(seconds=20), prefix_pad=timedelta(milliseconds=150), max_history_turns=12,
    ))
    await fake.client.join(ROOM, a)
    req = fake.joins[0]
    assert req.opening == pb.OPENING_TURN_AGENT and req.profile.opening == pb.OPENING_TURN_AGENT
    assert req.custom_fields_json == "" and req.participant_id == ""
    assert req.profile.ai_backend == "google-cascade" and req.profile.model == ""
    want = pb.CascadeConfig(
        llm_model="gemini-2.5-flash", llm_region="europe-west4", llm_thinking=True, llm_temperature=0.5,
        stt_model="short", stt_region="global", stt_languages=["ta-IN"], stt_keywords=["Ulai"],
        tts_voice="Charon", tts_region="asia-south1", tts_language="ta-IN", tts_rate=16000, vad="silero",
        vad_confidence=0.6, vad_min_volume=0.2, turn_model=False, turn_model_max_silence_ms=2000,
        transcript_wait_ms=300, endpoint_silence_ms=600, min_utterance_ms=200, max_utterance_ms=20000,
        prefix_pad_ms=150, max_history_turns=12,
    )
    assert req.profile.cascade == want, f"\n got: {req.profile.cascade}\nwant: {want}"


async def test_join_refuses_an_invalid_agent_before_the_server(fake):
    with pytest.raises(agent.ValidationError, match='unknown Agent.voice "Pukc"'):
        await fake.client.join(ROOM, support(voice="Pukc"))
    for room, want in [
        (agent.Room(id="", url="https://cp"), "join needs Room.id"),
        (agent.Room(id="r", url=""), "join needs Room.url"),
    ]:
        with pytest.raises(agent.ValidationError, match=want):
            await fake.client.join(room, support())
    with pytest.raises(agent.ValidationError, match="fields cannot be encoded"):
        await fake.client.join(ROOM, support(), fields={"when": object()})
    assert fake.joins == []


async def test_unlisted_values_for_platforms(fake):
    """A platform forwarding agents from its own database sends the stored
    values, and the server is the judge. Ranges and structure are still checked."""
    stored = support(voice="SomeNewVoice", language="hi-IN,en-IN", pipeline=agent.Realtime(model="gemini-live-next"))
    with pytest.raises(agent.ValidationError):
        await fake.client.join(ROOM, stored)
    await fake.client.join(ROOM, stored, allow_unlisted_values=True)
    p = fake.joins[0].profile
    assert (p.voice, p.language, p.model, p.ai_backend) == (
        "SomeNewVoice", "hi-IN,en-IN", "gemini-live-next", "gemini-live",
    )

    stored.pipeline = agent.Cascade(stt="long_v9", llm="gpt-x", vad="webrtc")
    await fake.client.join(ROOM, stored, allow_unlisted_values=True)
    c = fake.joins[1].profile.cascade
    assert (c.stt_model, c.stt_region, c.llm_model, c.vad) == ("long_v9", "", "gpt-x", "webrtc")

    # Still refused: values no server could take.
    stored.temperature = 9
    with pytest.raises(agent.ValidationError, match="outside 0 to 2"):
        await fake.client.join(ROOM, stored, allow_unlisted_values=True)
    stored.temperature, stored.opening = 0.3, "whoever"
    with pytest.raises(agent.ValidationError, match="unknown Agent.opening"):
        await fake.client.join(ROOM, stored, allow_unlisted_values=True)
    stored.opening, stored.voice = None, 42
    with pytest.raises(agent.ValidationError, match="must be a string"):
        await fake.client.join(ROOM, stored, allow_unlisted_values=True)
    assert len(fake.joins) == 2


# ── Secrets ───────────────────────────────────────────────────────────────────


def test_secrets_stay_out_of_logs():
    cloud = agent.CloudAccount(project="p", region="r", credentials_json='{"private_key":"SECRET"}')
    room = agent.Room(id="r1", url="https://cp", token="SECRET")
    a = support(cloud=cloud)
    for s in [repr(cloud), str(cloud), f"{cloud}", f"{cloud!r}", "%s" % (cloud,), repr(room), str(room),  # noqa: UP031
              f"{room}", repr(a), str(a), repr([cloud, room])]:
        assert "SECRET" not in s, s
    assert "r1" in str(room) and "[redacted]" in str(room)
    assert "[redacted]" in repr(cloud) and "'p'" in repr(cloud)


def test_cloud_account_from_file(tmp_path):
    good = tmp_path / "key.json"
    good.write_text(json.dumps({"type": "service_account", "project_id": "p"}))
    c = agent.CloudAccount.from_file("p", "us-central1", good)
    assert (c.project, c.region) == ("p", "us-central1") and "service_account" in c.credentials_json
    assert agent.CloudAccount.from_file("p", "", str(good)).region == ""

    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    with pytest.raises(agent.ValidationError, match="is not a service-account key file"):
        agent.CloudAccount.from_file("p", "", bad)
    untyped = tmp_path / "untyped.json"
    untyped.write_text("{}")
    with pytest.raises(agent.ValidationError, match="is not a service-account key file"):
        agent.CloudAccount.from_file("p", "", untyped)
    with pytest.raises(agent.ValidationError, match="reading the service-account key"):
        agent.CloudAccount.from_file("p", "", tmp_path / "missing.json")
