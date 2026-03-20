"""
1_context_write.py  –  WRITE: Context Creation and Token Tracking
=================================================================

Concept
-------
Every time you send a message to an LLM, the *entire* conversation
history is re-transmitted.  This demo builds a multi-turn conversation
turn-by-turn, printing a token-usage bar after every exchange so you
can *see* the context window filling up in real time.

How it works with Strands Agents
---------------------------------
Strands Agents maintains its own message list internally.  We access
``agent.messages`` after each call to count tokens on the live
conversation state.  We also register a ``token_tracker`` tool that
the agent may optionally call – but the real tracking happens in the
outer Python loop.

What to observe
---------------
- The context grows monotonically; nothing is ever discarded here.
- After ~8 turns the bar shifts from green → yellow → red.
- Compare these numbers with Demo 3 (COMPRESS) to see the difference
  that summarisation makes.
"""

import sys
import os
import json

# Make the project root importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strands import Agent
from strands.models import BedrockModel
from utils import TokenCounter, Visualizer

# ── Load config ────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

# ── Globals ────────────────────────────────────────────────────────────
CONTEXT_WINDOW = CFG.get("context_window_limit", 8000)
counter  = TokenCounter()
viz      = Visualizer(bar_width=CFG["demo_settings"].get("bar_width", 40))

# ── Conversation script ────────────────────────────────────────────────
CONVERSATION = [
    "Hi! Can you explain what machine learning is in one sentence?",
    "Great. Now explain supervised learning with a concrete example.",
    "What is the difference between classification and regression?",
    "Give me a quick overview of neural networks.",
    "How does backpropagation work?",
    "What is overfitting and how do we prevent it?",
    "Summarise everything we have discussed so far.",
    "One last question: what should I learn first if I am a beginner?",
]


def run_demo() -> None:
    viz.print_header(
        "DEMO 1 · WRITE",
        "Context Creation and Token Tracking",
    )

    model = BedrockModel(
        model_id=CFG["model_id"],
        region_name=CFG["aws_region"],
    )
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a concise ML tutor. "
            "Keep every answer under 80 words."
        ),
    )

    print(f"  Context window limit: {CONTEXT_WINDOW:,} tokens\n")

    for turn, user_msg in enumerate(CONVERSATION, 1):
        viz.print_section(f"Turn {turn} / {len(CONVERSATION)}")
        print(f"  {chr(0x1F9D1)} USER : {user_msg[:70]}...")

        # ── Call the agent ─────────────────────────────────────────────
        response = agent(user_msg)

        # ── Extract text safely ────────────────────────────────────────
        if hasattr(response, "message") and response.message:
            content = response.message.get("content", "")
            if isinstance(content, list):
                reply_text = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            else:
                reply_text = str(content)
        else:
            reply_text = str(response)

        reply_preview = reply_text[:120].replace("\n", " ")
        print(f"  🤖 AGENT: {reply_preview}...")

        # ── Count tokens on the live message list ──────────────────────
        messages = agent.messages or []
        total_tokens = counter.count_messages(messages)
        pct = total_tokens / CONTEXT_WINDOW * 100

        viz.print_token_bar(f"Context (turn {turn})", total_tokens, CONTEXT_WINDOW)

        # ── Context pressure warnings ──────────────────────────────────
        if pct > 85:
            viz.print_warning("⚠  Context > 85 % — consider COMPRESS or SELECT!")
        elif pct > 60:
            viz.print_warning("Context > 60 % — approaching compression threshold.")

    # ── Final breakdown ────────────────────────────────────────────────
    breakdown = counter.get_message_breakdown(agent.messages or [])
    if breakdown:
        viz.print_message_table(breakdown, CONTEXT_WINDOW)

    viz.print_kv(
        "Demo 1 Summary",
        {
            "Total turns": str(len(CONVERSATION)),
            "Final token count": f"{counter.count_messages(agent.messages or []):,}",
            "Context window": f"{CONTEXT_WINDOW:,}",
            "Utilisation": f"{counter.count_messages(agent.messages or []) / CONTEXT_WINDOW * 100:.1f} %",
            "Key takeaway": "Context grows with every turn — plan accordingly",
        },
    )


if __name__ == "__main__":
    run_demo()
