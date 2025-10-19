## Local ChatGPT Agent Example

This repository contains a minimal example of how to build and run a local
ChatGPT-style agent using the OpenAI Web API. The single-agent implementation
lives in [`agent.py`](agent.py) and demonstrates how to:

* Create a reusable agent class that keeps conversation state.
* Call the [`responses.create`](https://platform.openai.com/docs/guides/responses) endpoint with the official
  `openai` Python SDK.
* Run the agent either with a one-off prompt or in an interactive chat loop.

### Prerequisites

1. Python 3.9 or newer.
2. Install dependencies:

   ```bash
   pip install openai python-dotenv
   ```

3. Set your OpenAI API key. The easiest way during development is to create a
   `.env` file alongside `agent.py`:

   ```dotenv
   OPENAI_API_KEY=sk-your-key-here
   ```

   Alternatively, export `OPENAI_API_KEY` in your shell.

### Usage

Run a single-turn request:

```bash
python agent.py "Write a short haiku about local agents."
```

Start an interactive chat session:

```bash
python agent.py --interactive
```

You can customise the agent by passing flags:

```bash
python agent.py --model gpt-4o-mini --system "You are a cybersecurity expert." "How can I secure my web app?"
```

Refer to the docstrings inside `agent.py` for more details.

### Multi-agent conversations

For larger orchestrations you can run the collaborative multi-agent driver in
[`multi_agent.py`](multi_agent.py). It spins up a group of `ChatGPTAgent`
instances, feeds them the shared transcript, and coordinates a round-robin
discussion. Each agent is defined with the `--agent` flag using the
`name@model:"system prompt"` syntax (the `@model` segment is optional):

```bash
python multi_agent.py \
  --topic "Plan a product launch" \
  --agent strategist:"Outline the go-to-market strategy." \
  --agent marketer@gpt-4o-mini:"Draft messaging for social media." \
  --agent engineer:"Identify technical dependencies." \
  --rounds 2 \
  --save-transcript transcripts/launch.json
```

The script prints every message as it is generated and optionally writes the
entire transcript to disk.
