"""Minimal local ChatGPT-style agent using the OpenAI Web API.

The `ChatGPTAgent` class wraps the official `openai` Python SDK and exposes a
simple API for building stateful conversations. It can be used as a building
block in larger applications or invoked directly from the command line.

Example:

    from agent import ChatGPTAgent

    agent = ChatGPTAgent(system_prompt="You are a helpful assistant.")
    response = agent.ask("How do I write a CLI around the OpenAI API?")
    print(response)

Run ``python agent.py --help`` for the available CLI options.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Iterable, List, MutableSequence, Optional

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from a local .env file if present. This is a
# no-op when the file does not exist and keeps configuration lightweight.
load_dotenv()


MessageContent = List[dict]


def _text_content(text: str) -> MessageContent:
    """Represent plain text in the format expected by the Responses API."""

    return [{"type": "text", "text": text}]


@dataclass
class ChatGPTAgent:
    """A conversational agent backed by the OpenAI Responses API.

    Parameters
    ----------
    model:
        The model identifier to use. Defaults to ``"gpt-4o-mini"``.
    system_prompt:
        Optional system prompt injected at the beginning of the conversation.
    client:
        Optional OpenAI client instance. When omitted, a new ``OpenAI`` client
        will be created using the standard environment variables.
    history:
        Optional iterable of past messages used to seed the agent's memory.
    """

    model: str = "gpt-4o-mini"
    system_prompt: Optional[str] = "You are a helpful assistant."
    client: Optional[OpenAI] = None
    history: Iterable[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or add it to a .env file."
            )

        self.client = self.client or OpenAI()

        # Start the message history. The Responses API expects the "content"
        # field to be a list of typed blocks; in this example we only use text.
        self._messages: MutableSequence[dict] = []

        if self.system_prompt:
            self._messages.append(
                {"role": "system", "content": _text_content(self.system_prompt)}
            )

        for message in self.history:
            self._messages.append(self._normalise_message(message))

    @staticmethod
    def _normalise_message(message: dict) -> dict:
        """Ensure message dictionaries conform to the Responses API schema."""

        if "role" not in message:
            raise ValueError("Conversation messages must define a role.")

        content = message.get("content")
        if isinstance(content, str):
            content = _text_content(content)

        if not isinstance(content, list):
            raise ValueError("Message content must be a string or list of blocks.")

        return {"role": message["role"], "content": content}

    def ask(self, prompt: str, temperature: float = 0.7) -> str:
        """Send a prompt to the agent and return the assistant's reply."""

        self._messages.append({"role": "user", "content": _text_content(prompt)})

        response = self.client.responses.create(
            model=self.model,
            input=list(self._messages),
            temperature=temperature,
        )

        reply = response.output_text

        self._messages.append(
            {"role": "assistant", "content": _text_content(reply)}
        )

        return reply

    def reset(self) -> None:
        """Clear the conversation history, preserving the system prompt."""

        self._messages = []
        if self.system_prompt:
            self._messages.append(
                {"role": "system", "content": _text_content(self.system_prompt)}
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional prompt to send immediately after creating the agent.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt to prime the agent.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for responses (default: %(default)s)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive REPL instead of running a single prompt.",
    )
    return parser


def _run_interactive(agent: ChatGPTAgent, temperature: float) -> None:
    print("Starting interactive session. Type 'exit' or Ctrl-D to quit.")

    try:
        while True:
            try:
                user_input = input("you> ")
            except EOFError:
                print()  # Add a newline after Ctrl-D
                break

            if user_input.strip().lower() in {"exit", "quit"}:
                break

            if not user_input.strip():
                continue

            reply = agent.ask(user_input, temperature=temperature)
            print(f"assistant> {reply}")
    except KeyboardInterrupt:
        print("\nSession terminated.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    agent = ChatGPTAgent(model=args.model, system_prompt=args.system)

    if args.interactive:
        _run_interactive(agent, temperature=args.temperature)
        return 0

    if not args.prompt:
        parser.error("Provide a prompt argument or use --interactive.")

    prompt = " ".join(args.prompt)
    reply = agent.ask(prompt, temperature=args.temperature)
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
