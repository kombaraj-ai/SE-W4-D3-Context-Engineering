"""
3_context_compress.py  –  COMPRESS: Context Compression
========================================================

Concept
-------
Even after selective filtering, long conversations can exceed the
context window.  COMPRESS summarises older turns into a compact
"rolling summary" message, replacing many tokens with far fewer
while retaining the semantic core.

Two compression modes are demonstrated:

  Hard-coded   – we author a summary by hand (shows the *ideal* result).
  Agent-based  – we ask a Strands Agent to compress the history itself
                 using a dedicated ``summarise_conversation`` tool.

How it works with Strands Agents
---------------------------------
The compression agent receives the raw message list serialised as JSON
and is instructed to return a single-paragraph summary.  That summary
is then injected as a ``{"role": "system", "content": "<summary>"}``
message at the head of a new, slim history.

The compressed history is then used for a follow-up query, exactly as
in Demo 2.

What to observe
---------------
- Token counts before and after compression.
- The compression ratio and percentage saved.
- The agent's answer quality remains high despite the shorter context.
"""

import sys
import os
import json
from typing import List, Dict, Any, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strands import Agent, tool
from strands.models import BedrockModel
from utils import TokenCounter, Visualizer

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

CONTEXT_WINDOW = CFG.get("context_window_limit", 8000)
counter  = TokenCounter()
viz      = Visualizer(bar_width=CFG["demo_settings"].get("bar_width", 40))


# ── Content format helpers ─────────────────────────────────────────────

def _block(text: str) -> List[Dict[str, str]]:
    """Wrap a plain string in Strands' required content-block format."""
    return [{"type": "text", "text": text}]


def _text(content: Union[str, list, None]) -> str:
    """Extract plain text from any Strands content format (block list or string)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


# ── Long conversation to compress ─────────────────────────────────────
#
# All content values use Strands' block format: [{"type": "text", "text": "..."}]
# Passing plain strings causes: TypeError: content_type=<[> | unsupported type
#
LONG_HISTORY: List[Dict] = [
    {"role": "user",      "content": _block("I want to build a web application for task management. Where should I start?")},
    {"role": "assistant", "content": _block("Start by defining your requirements: user authentication, task CRUD, deadlines, and collaboration. Choose a tech stack — React + FastAPI is popular.")},
    {"role": "user",      "content": _block("Let's go with React for frontend. What state management should I use?")},
    {"role": "assistant", "content": _block("For a mid-size app, Zustand or Redux Toolkit are great. Zustand has a simpler API; Redux is better for complex state trees.")},
    {"role": "user",      "content": _block("OK. For the backend, should I use FastAPI or Django?")},
    {"role": "assistant", "content": _block("FastAPI is excellent for async REST/GraphQL APIs with auto-generated docs. Django REST Framework offers more batteries but is heavier.")},
    {"role": "user",      "content": _block("Which database would you recommend?")},
    {"role": "assistant", "content": _block("PostgreSQL for relational data. SQLite for local dev. If you need real-time updates, add Redis for pub/sub or WebSockets.")},
    {"role": "user",      "content": _block("What about authentication?")},
    {"role": "assistant", "content": _block("JWT tokens with refresh logic are standard. Libraries: FastAPI-Users or Auth0 for managed OAuth2/OIDC.")},
    {"role": "user",      "content": _block("How should I deploy this?")},
    {"role": "assistant", "content": _block("Containerise with Docker, orchestrate with Docker Compose for dev. For prod, ECS/Fargate on AWS or a Kubernetes cluster. Add a CDN (CloudFront) for the React bundle.")},
    {"role": "user",      "content": _block("What CI/CD pipeline do you suggest?")},
    {"role": "assistant", "content": _block("GitHub Actions: lint → test → build Docker image → push to ECR → deploy via ECS rolling update. Add staging and prod environments.")},
    {"role": "user",      "content": _block("Any monitoring recommendations?")},
    {"role": "assistant", "content": _block("Datadog or Grafana + Prometheus for metrics. Sentry for error tracking. AWS CloudWatch for infra-level logs.")},
]


# ── Compression tool ───────────────────────────────────────────────────

@tool
def summarise_conversation(messages_json: str) -> str:
    """
    Summarise a JSON-encoded list of chat messages into a compact paragraph.

    Parameters
    ----------
    messages_json : str
        JSON string of message dicts with 'role' and 'content' keys.

    Returns
    -------
    str
        A concise summary (≤ 150 words) capturing all key decisions.
    """
    # This tool body is intentionally thin – the LLM handles the summarisation
    # by interpreting the returned string as instruction context.
    return (
        "Please produce a ≤150-word summary of the conversation "
        "capturing every technology choice and decision made. "
        "Return only the summary paragraph, no headings.\n\n"
        f"Messages:\n{messages_json}"
    )


def compress_with_agent(
    history: List[Dict],
    model: BedrockModel,
) -> str:
    """Ask a Strands agent to summarise *history* and return the summary string."""
    compress_agent = Agent(
        model=model,
        system_prompt=(
            "You are a conversation summariser. "
            "When given a JSON message list, return ONLY a single dense paragraph "
            "capturing all key facts, decisions, and technology choices. "
            "Maximum 150 words."
        ),
        tools=[summarise_conversation],
    )

    # Flatten content blocks to plain strings before serialising to JSON.
    # The compression prompt goes to the LLM as readable text — it should
    # not contain nested block dicts like [{"type":"text","text":"..."}].
    flat_history = [
        {"role": m["role"], "content": _text(m["content"])}
        for m in history
    ]
    history_json = json.dumps(flat_history, indent=2)
    response = compress_agent(f"Summarise this conversation history:\n\n{history_json}")

    if hasattr(response, "message") and response.message:
        content = response.message.get("content", "")
        if isinstance(content, list):
            return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        return str(content)
    return str(response)


def build_compressed_history(summary: str) -> List[Dict]:
    """
    Wrap the summary as a 2-message compressed history.

    Content values use Strands' block format so Agent(messages=...) accepts them.
    """
    return [
        {
            "role": "user",
            "content": _block(
                f"[CONVERSATION SUMMARY]\n{summary}\n"
                "[END SUMMARY – continue from here]"
            ),
        },
        {
            "role": "assistant",
            "content": _block(
                "Understood. I have the context from the summary and am ready to continue."
            ),
        },
    ]


def run_demo() -> None:
    viz.print_header(
        "DEMO 3 · COMPRESS",
        "Context Compression via Summarisation",
    )

    model = BedrockModel(
        model_id=CFG["model_id"],
        region_name=CFG["aws_region"],
    )

    # ── Step 1: show baseline token cost ──────────────────────────────
    viz.print_section("Step 1 – Baseline (uncompressed)")
    orig_tokens = counter.count_messages(LONG_HISTORY)
    viz.print_token_bar("Full history", orig_tokens, CONTEXT_WINDOW)
    print(f"  {len(LONG_HISTORY)} messages occupying {orig_tokens:,} tokens.\n")

    # ── Step 2: agent-based compression ───────────────────────────────
    viz.print_section("Step 2 – Compressing with Strands Agent")
    print("  Asking compression agent to summarise the conversation …")
    summary = compress_with_agent(LONG_HISTORY, model)
    print(f"\n  Summary produced ({len(summary)} chars):")
    print(f"  ┌─\n  │ {summary[:300].replace(chr(10), chr(10) + '  │ ')}\n  └─\n")

    compressed_history = build_compressed_history(summary)
    stats = counter.calculate_compression_ratio(LONG_HISTORY, compressed_history)
    viz.print_compression_stats(stats)

    # ── Step 3: continue conversation with compressed context ──────────
    viz.print_section("Step 3 – Follow-up query on compressed context")
    follow_up = "Given everything we decided, what should I tackle first this week?"
    print(f"  User: {follow_up}\n")

    agent = Agent(
        model=model,
        system_prompt="You are a helpful software architecture advisor.",
        messages=compressed_history,
    )
    response = agent(follow_up)

    if hasattr(response, "message") and response.message:
        content = response.message.get("content", "")
        if isinstance(content, list):
            reply = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        else:
            reply = str(content)
    else:
        reply = str(response)

    print(f"  🤖 Agent: {reply[:300].replace(chr(10), ' ')}...")

    # ── Summary ────────────────────────────────────────────────────────
    viz.print_kv(
        "Demo 3 Summary",
        {
            "Original messages": str(len(LONG_HISTORY)),
            "Compressed messages": str(len(compressed_history)),
            "Tokens saved": f"{stats['saved_tokens']:,}",
            "Space freed": f"{stats['percentage_saved']:.1f} %",
            "Key insight": "Summaries preserve semantics; token cost drops sharply",
        },
    )


if __name__ == "__main__":
    run_demo()
