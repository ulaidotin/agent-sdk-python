"""A complete voice agent in one file.

    python examples/basic.py --mode join --room <room-id> --room-url <url> --room-token <token>
    python examples/basic.py --mode serve

The server address and key come from ULAI_ADDR and ULAI_API_KEY; the Google
Cloud project the models run in from GOOGLE_CLOUD_PROJECT and
GOOGLE_APPLICATION_CREDENTIALS (a service-account key file).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Annotated

import ulai_agents as agent

log = logging.getLogger("basic")


@agent.tool(description="Look up the balance of an account.")
async def check_balance(account: Annotated[str, "The account number."]) -> dict:
    """The tool's parameters are read off this signature."""
    return {"account": account, "balance_rupees": 250}


# support is the agent: who it is, how it sounds, how it hears and speaks, and
# what it can do.
support = agent.Agent(
    id="support",
    instructions="You are a bank's support agent. Be brief. Use check_balance when asked for a balance.",
    greeting="Hello, how can I help you today?",
    closing="Thanks for calling. Goodbye.",
    language=agent.Language.ENGLISH_INDIA,
    voice=agent.Voice.KORE,
    # One speech-to-speech model. For three separate stages instead:
    #   agent.Cascade(llm=agent.LLMModel.GEMINI_2_5_FLASH, stt=agent.STTModel.SHORT, vad=agent.VAD.SILERO)
    pipeline=agent.Realtime(model=agent.RealtimeModel.GEMINI_LIVE_2_5_FLASH_NATIVE_AUDIO),
    tools=[check_balance],
)


async def handle(session: agent.Session) -> None:
    """The agent's own logic for one call."""
    log.info("call %s from %s", session.id, session.call.from_)

    async for event in session.events():
        match event:
            case agent.Transcript(final=True, speaker=speaker, text=text):
                log.info("%s: %s", speaker, text)
                # Steer the model when the caller asks for something only you
                # can decide.
                if speaker == agent.Speaker.CALLER and "manager" in text.lower():
                    try:
                        await session.reply("Tell the caller a manager will call them back within an hour.")
                    except agent.AgentError as e:
                        log.warning("reply: %s", e)
            case agent.Rejected(command=command, message=message):
                log.warning("the server refused %s: %s", command, message)
            case agent.Ended(reason=reason, heard=heard):
                log.info("call ended: %s, caller heard %.1fs", reason, heard.total_seconds())


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="join", choices=["join", "serve"],
                        help="join: start a call in a room; serve: take calls as they arrive")
    parser.add_argument("--room", default="", help="join: the room")
    parser.add_argument("--room-url", default="", help="join: the control plane holding the room")
    parser.add_argument("--room-token", default="", help="join: the token for that control plane")
    args = parser.parse_args()

    # The server holds no model credentials: each agent brings its own.
    support.cloud = agent.CloudAccount.from_file(
        os.environ.get("GOOGLE_CLOUD_PROJECT", ""), "", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    )

    # Plaintext is allowed only to this machine; pass tls=True for a server
    # anywhere else.
    address = os.environ.get("ULAI_ADDR") or "127.0.0.1:50052"
    async with agent.Client(address, api_key=os.environ.get("ULAI_API_KEY")) as client:
        if args.mode == "join":
            room = agent.Room(id=args.room, url=args.room_url, token=args.room_token)
            await client.run(room, support, handle)
        else:
            log.info("waiting for calls to %s", support.id)
            await client.serve(support.id, support.tools, handle)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Ctrl-C: run has taken the agent out of the room
    except agent.AgentError as e:
        log.error("%s", e)
        sys.exit(1)
