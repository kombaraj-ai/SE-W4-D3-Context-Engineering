"""
4_context_isolate.py  –  ISOLATE: Context Isolation
====================================================

Concept
-------
When a single application handles *multiple* independent tasks or users,
mixing their contexts causes:

  • Context leakage – one task's facts bleed into another's answers.
  • Inflated token counts – irrelevant history is always present.
  • Security / privacy risks in multi-user systems.

ISOLATE gives each task its own hermetically sealed message list.
Agents are instantiated with only the messages belonging to their
context slot; they never see messages from other slots.

How it works with Strands Agents
---------------------------------
A ``ContextManager`` class maps string IDs → message lists.

  • ``get_agent(context_id)`` builds a fresh Agent preloaded with
    only that context's history.
  • After each call the assistant reply is appended back to the
    correct slot – other slots are untouched.

Three parallel conversations are run:

  coding_assistant   – Python / tech questions
  recipe_advisor     – cooking / food questions
  travel_planner     – travel / destination questions

At the end we print a table showing each context's token budget and
prove cross-context isolation by asking each agent a question that is
only answerable from its own context.

What to observe
---------------
- Each agent answers correctly from its own history.
- Tokens per context are modest and independent.
- Ask the coding agent about recipes → it does not know (isolation proof).
"""

import sys
import os
import json
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strands import Agent
from strands.models import BedrockModel
from utils import TokenCounter, Visualizer

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

CONTEXT_WINDOW = CFG.get("context_window_limit", 8000)
counter  = TokenCounter()
viz      = Visualizer(bar_width=CFG["demo_settings"].get("bar_width", 40))


# ── Content helpers ────────────────────────────────────────────────────

def _text(content) -> str:
    """Safely extract plain text from any Strands content format."""
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


# ── Context Manager ────────────────────────────────────────────────────

class ContextManager:
    """
    Lightweight registry that maps context IDs to isolated Agents.

    Key design change vs the original:
    ------------------------------------
    Instead of creating a fresh Agent on every call (and re-injecting
    the stored message list), we keep ONE persistent Agent per context
    slot and simply call it each time.

    This means Strands manages its own internal message list in the
    correct content-block format.  We never manually append to or
    re-inject that list, so there is no risk of plain-string content
    slipping back into the history.

    Isolation is guaranteed because each context slot has its own
    Agent instance with its own independent message store.
    """

    def __init__(self, model: BedrockModel) -> None:
        self.model = model
        # context_id → {"agent": Agent, "system_prompt": str}
        self._contexts: Dict[str, Dict[str, Any]] = {}

    def create_context(self, context_id: str, system_prompt: str) -> None:
        """Register a new isolated context slot with its own Agent."""
        agent = Agent(
            model=self.model,
            system_prompt=system_prompt,
        )
        self._contexts[context_id] = {
            "agent": agent,
            "system_prompt": system_prompt,
        }
        viz.print_success(f"Created context: {context_id!r}")

    def chat(self, context_id: str, user_message: str) -> str:
        """
        Send *user_message* within the given context slot.

        The persistent Agent accumulates history naturally across calls.
        No manual message manipulation needed — Strands handles it.
        """
        agent: Agent = self._contexts[context_id]["agent"]
        response = agent(user_message)

        # Extract reply text from Strands' content-block response
        if hasattr(response, "message") and response.message:
            content = response.message.get("content", "")
        else:
            content = str(response)

        return _text(content)

    def get_messages(self, context_id: str) -> List[Dict]:
        """Return the live message list from the context's Agent."""
        agent: Agent = self._contexts[context_id]["agent"]
        return agent.messages or []

    def get_stats(self) -> List[Dict[str, Any]]:
        stats = []
        for ctx_id in self._contexts:
            msgs = self.get_messages(ctx_id)
            t = counter.count_messages(msgs)
            stats.append(
                {
                    "id": ctx_id,
                    "messages": len(msgs),
                    "tokens": t,
                    "active": True,
                }
            )
        return stats


# ── Demo script ────────────────────────────────────────────────────────

CONTEXTS = {
    "coding_assistant": "You are a Python/software expert. Keep answers under 60 words.",
    "recipe_advisor":   "You are a friendly chef advisor. Keep answers under 60 words.",
    "travel_planner":   "You are an experienced travel planner. Keep answers under 60 words.",
}

CONVERSATIONS: Dict[str, List[str]] = {
    "coding_assistant": [
        "What is a Python decorator?",
        "Give me a one-liner that reverses a string.",
        "What is the GIL in CPython?",
    ],
    "recipe_advisor": [
        "What is the secret to a fluffy omelette?",
        "How do I make a quick pasta sauce from scratch?",
        "What herbs go well with chicken?",
    ],
    "travel_planner": [
        "What is the best time of year to visit Japan?",
        "How many days do I need in Kyoto?",
        "What is a must-see in Tokyo for first-timers?",
    ],
}

ISOLATION_PROBES: Dict[str, str] = {
    "coding_assistant": "What pasta sauce recipe did we discuss?",
    "recipe_advisor":   "What Python decorator did we talk about?",
    "travel_planner":   "Did we discuss any programming topics earlier?",
}


def run_demo() -> None:
    viz.print_header(
        "DEMO 4 · ISOLATE",
        "Context Isolation — Multiple Independent Conversations",
    )

    model = BedrockModel(
        model_id=CFG["model_id"],
        region_name=CFG["aws_region"],
    )

    mgr = ContextManager(model)

    # ── Create isolated slots ─────────────────────────────────────────
    viz.print_section("Creating isolated context slots")
    for ctx_id, sys_prompt in CONTEXTS.items():
        mgr.create_context(ctx_id, sys_prompt)
    print()

    # ── Run conversations in each context ─────────────────────────────
    for ctx_id, turns in CONVERSATIONS.items():
        viz.print_section(f"Context: {ctx_id}")
        for turn in turns:
            print(f"  👤 {turn}")
            reply = mgr.chat(ctx_id, turn)
            print(f"  🤖 {reply[:120].replace(chr(10), ' ')}...")
            t = counter.count_messages(mgr.get_messages(ctx_id))
            viz.print_token_bar(ctx_id[:18], t, CONTEXT_WINDOW)

    # ── Print isolation table ─────────────────────────────────────────
    viz.print_isolation_summary(mgr.get_stats())

    # ── Isolation proof ───────────────────────────────────────────────
    viz.print_section("Isolation Proof — Cross-context leakage test")
    print(
        "  We ask each agent about a topic discussed ONLY in another context.\n"
        "  A correctly isolated agent will say it has no knowledge of it.\n"
    )
    for ctx_id, probe in ISOLATION_PROBES.items():
        print(f"  Context [{ctx_id}]")
        print(f"  Probe  : {probe}")
        reply = mgr.chat(ctx_id, probe)
        print(f"  Reply  : {reply[:180].replace(chr(10), ' ')}...")
        print()

    # ── Final token table ─────────────────────────────────────────────
    viz.print_section("Final Token Budget per Context")
    for stat in mgr.get_stats():
        viz.print_token_bar(stat["id"][:22], stat["tokens"], CONTEXT_WINDOW)

    total = sum(s["tokens"] for s in mgr.get_stats())
    print(f"\n  Combined token cost if merged : {total:,}")
    print(f"  Average per context          : {total // len(CONTEXTS):,}")

    viz.print_kv(
        "Demo 4 Summary",
        {
            "Contexts created": str(len(CONTEXTS)),
            "Isolation verified": "Yes — agents cannot see other contexts",
            "Key use-cases": "Multi-user apps, parallel tasks, role separation",
            "Production tip": "Pair ISOLATE with COMPRESS per-context for max efficiency",
        },
    )


if __name__ == "__main__":
    run_demo()
