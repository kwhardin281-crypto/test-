"""Run collaborative conversations between multiple ChatGPT-style agents.

This module builds on :class:`agent.ChatGPTAgent` to orchestrate a round-robin
conversation between several roles. Each agent receives the shared transcript
and produces its next contribution. The orchestrator prints every response to
stdout and optionally saves the final transcript to disk.

Example usage::

    python multi_agent.py \
        --topic "Design a secure authentication flow" \
        --agent researcher@gpt-4o-mini:"You draft detailed proposals." \
        --agent critic:"You highlight potential flaws." \
        --rounds 3

The ``name@model:"system prompt"`` syntax for ``--agent`` entries lets you set a
per-agent model. When the ``@model`` segment is omitted, the default model set
with ``--model`` is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, MutableSequence, Optional, Sequence

from agent import ChatGPTAgent


@dataclass
class AgentConfig:
    """Configuration describing how to instantiate a conversational agent."""

    name: str
    system_prompt: str
    model: Optional[str] = None


class MultiAgentOrchestrator:
    """Coordinate a multi-agent conversation using :class:`ChatGPTAgent`."""

    def __init__(
        self,
        agents: Sequence[AgentConfig],
        *,
        topic: str,
        default_model: str,
        temperature: float,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("Provide at least two agents to start a conversation.")

        self.topic = topic
        self.temperature = temperature
        self._transcript: MutableSequence[dict] = [
            {"speaker": "moderator", "message": f"Topic: {topic}"}
        ]
        self._participants: List[tuple[AgentConfig, ChatGPTAgent]] = []

        for config in agents:
            model = config.model or default_model
            self._participants.append(
                (
                    config,
                    ChatGPTAgent(model=model, system_prompt=config.system_prompt),
                )
            )

    @property
    def transcript(self) -> List[dict]:
        """Return a copy of the transcript collected so far."""

        return list(self._transcript)

    def run(self, rounds: int) -> Iterator[dict]:
        """Execute the conversation and yield each generated message.

        Parameters
        ----------
        rounds:
            Number of conversation rounds. Each agent speaks exactly once per
            round in the order they were supplied.
        """

        if rounds < 1:
            raise ValueError("The number of rounds must be at least 1.")

        for round_index in range(rounds):
            for config, agent in self._participants:
                agent.reset()
                prompt = self._build_prompt(config.name)
                reply = agent.ask(prompt, temperature=self.temperature).strip()
                message = {
                    "round": round_index + 1,
                    "speaker": config.name,
                    "message": reply,
                }
                self._transcript.append(
                    {"speaker": config.name, "message": reply}
                )
                yield message

    def _build_prompt(self, speaker: str) -> str:
        """Render the prompt sent to the next agent."""

        history_lines = [
            f"{entry['speaker']}: {entry['message']}" for entry in self._transcript
        ]
        history = "\n".join(history_lines)
        return (
            "You are participating in a collaborative discussion.\n"
            f"Your role: {speaker}.\n"
            f"Shared topic: {self.topic}.\n\n"
            "Conversation so far:\n"
            f"{history}\n\n"
            "Write your next contribution in a concise paragraph."
        )


def _parse_agent_entry(entry: str) -> AgentConfig:
    """Parse ``--agent`` CLI entries.

    The format is ``name@model:system prompt`` where ``@model`` is optional.
    """

    if ":" not in entry:
        raise argparse.ArgumentTypeError(
            "Agent definitions must use the 'name@model:system prompt' format."
        )

    header, system_prompt = entry.split(":", 1)
    if not system_prompt.strip():
        raise argparse.ArgumentTypeError("System prompt text cannot be empty.")

    if "@" in header:
        name, model = header.split("@", 1)
        name = name.strip()
        model = model.strip()
        model_value: Optional[str] = model or None
    else:
        name = header.strip()
        model_value = None

    if not name:
        raise argparse.ArgumentTypeError("Agent names cannot be empty.")

    return AgentConfig(name=name, system_prompt=system_prompt.strip(), model=model_value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        required=True,
        help="High-level topic shared with all agents.",
    )
    parser.add_argument(
        "--agent",
        action="append",
        required=True,
        type=_parse_agent_entry,
        help="Define an agent using the 'name@model:system prompt' syntax.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rounds to run (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Default model for agents without an explicit model override.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature applied to every agent (default: %(default)s)",
    )
    parser.add_argument(
        "--save-transcript",
        type=Path,
        help="Optional path to store the transcript as JSON.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    orchestrator = MultiAgentOrchestrator(
        args.agent,
        topic=args.topic,
        default_model=args.model,
        temperature=args.temperature,
    )

    try:
        for message in orchestrator.run(args.rounds):
            print(f"[{message['round']}|{message['speaker']}] {message['message']}")
    except KeyboardInterrupt:
        print("\nConversation interrupted.")

    if args.save_transcript:
        args.save_transcript.parent.mkdir(parents=True, exist_ok=True)
        with args.save_transcript.open("w", encoding="utf-8") as fh:
            json.dump(orchestrator.transcript, fh, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
