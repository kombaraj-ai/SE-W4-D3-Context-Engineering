"""
2_context_select.py  –  SELECT: Selective Context Passing
=========================================================

Concept
-------
Not every past message is relevant to the *current* question.
Blindly forwarding the full history wastes tokens, inflates latency,
and can actually confuse the model (stale context "noise").

SELECT filters the conversation to only the messages that matter:

  • The system prompt is always kept.
  • The most recent N turns are always kept (recency window).
  • Older turns are kept only if their content matches a relevance
    keyword list derived from the current query.

How it works with Strands Agents
---------------------------------
Strands does not expose a hook to mutate the message list before each
call, so we manage the message list manually:

  1. We keep a canonical ``history`` list ourselves.
  2. Before each call we build a ``filtered`` sub-list with
     ``select_relevant_messages()``.
  3. We construct a *fresh* ``Agent`` per call with ``messages``
     pre-loaded to the filtered list.
  4. After the call we append the new user+assistant pair to the
     canonical history (unfiltered) for future turns.

This pattern is transparent to the end-user: the agent still responds
correctly, but token usage is dramatically lower.

What to observe
---------------
- "Before" vs "After" token counts in the printed bars.
- How many messages were dropped.
- The agent still gives a sensible answer because the most relevant
  turns were retained.
"""

import sys
import os
import re
import json
from typing import List, Dict, Any

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

# ── Helper: build a Strands-compatible content block ──────────────────
def _block(text: str) -> List[Dict[str, str]]:
    """Wrap a plain string in the content-block list Strands expects."""
    return [{"type": "text", "text": text}]


def _text(content) -> str:
    """
    Extract plain text from any content format:
      • list of Strands blocks  →  join all 'text' fields
      • plain string            →  return as-is
    """
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content) if content else ""


# ── Simulated multi-topic history (Strands content-block format) ───────
#
# Strands' Bedrock formatter requires content to be a list of typed
# blocks: [{"type": "text", "text": "..."}]
# Passing plain strings causes: TypeError: content_type=<W> | unsupported type
#
SEED_HISTORY: List[Dict] = [
    {"role": "user",      "content": _block("What is Python?")},
    {"role": "assistant", "content": _block("Python is a high-level, interpreted programming language known for its readability and simplicity.")},
    {"role": "user",      "content": _block("How do I install Python on Windows?")},
    {"role": "assistant", "content": _block("Download the installer from python.org, run it, and check 'Add Python to PATH'. Then verify with `python --version`.")},
    {"role": "user",      "content": _block("What is machine learning?")},
    {"role": "assistant", "content": _block("Machine learning is a subset of AI that enables systems to learn from data and improve from experience without explicit programming.")},
    {"role": "user",      "content": _block("What frameworks are popular for ML?")},
    {"role": "assistant", "content": _block("Popular ML frameworks include TensorFlow, PyTorch, scikit-learn, and Keras.")},
    {"role": "user",      "content": _block("Tell me about French cuisine.")},
    {"role": "assistant", "content": _block("French cuisine is renowned for its use of butter, cream, and wine. Classic dishes include coq au vin, bouillabaisse, and crème brûlée.")},
    {"role": "user",      "content": _block("What are the best French wines?")},
    {"role": "assistant", "content": _block("Bordeaux, Burgundy, and Champagne are among the most celebrated French wine regions.")},
    {"role": "user",      "content": _block("How does neural network training work?")},
    {"role": "assistant", "content": _block("Training involves forward passes to compute predictions, loss calculation, and backpropagation to update weights via gradient descent.")},
]

RECENCY_WINDOW = 2   # always keep the last N user+assistant pairs


def extract_keywords(query: str) -> List[str]:
    """Naïve keyword extractor: lower-case words longer than 3 chars."""
    words = re.findall(r"\b[a-zA-Z]{4,}\b", query.lower())
    STOP = {"what", "this", "that", "with", "have", "from", "they", "will",
            "about", "which", "there", "their", "been", "were", "when"}
    return [w for w in words if w not in STOP]


def select_relevant_messages(
    history: List[Dict],
    query: str,
    recency: int = RECENCY_WINDOW,
) -> List[Dict]:
    """
    Return a filtered message list for *query*.

    Strategy
    --------
    1. Always include the last ``recency`` user/assistant pairs.
    2. For older pairs, include only if any keyword from query appears
       in either the user or assistant message.
    """
    if not history:
        return []

    keywords = extract_keywords(query)

    # Group into (user_msg, assistant_msg) pairs
    pairs: List[tuple] = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            pairs.append((history[i], history[i + 1]))
            i += 2
        else:
            i += 1

    recent_pairs = pairs[-recency:]
    older_pairs  = pairs[:-recency] if len(pairs) > recency else []

    selected: List[Dict] = []

    for user_msg, asst_msg in older_pairs:
        # Use _text() to safely extract string from content blocks
        combined = (_text(user_msg["content"]) + " " + _text(asst_msg["content"])).lower()
        if any(kw in combined for kw in keywords):
            selected.extend([user_msg, asst_msg])

    for pair in recent_pairs:
        selected.extend(pair)

    return selected


def run_demo() -> None:
    viz.print_header(
        "DEMO 2 · SELECT",
        "Selective Context Passing",
    )

    model = BedrockModel(
        model_id=CFG["model_id"],
        region_name=CFG["aws_region"],
    )

    queries = [
        "Can you explain how PyTorch compares to TensorFlow for deep learning?",
        "What Python version should I use for data science projects?",
        "What makes Bordeaux wines special compared to other regions?",
    ]

    for q_idx, query in enumerate(queries, 1):
        viz.print_section(f"Query {q_idx}: {query[:60]}...")

        # ── Full history tokens ────────────────────────────────────────
        full_tokens = counter.count_messages(SEED_HISTORY)

        # ── Filtered history ───────────────────────────────────────────
        filtered = select_relevant_messages(SEED_HISTORY, query)
        filtered_tokens = counter.count_messages(filtered) if filtered else 0

        viz.print_select_summary(
            total=len(SEED_HISTORY),
            selected=len(filtered),
            token_before=full_tokens,
            token_after=filtered_tokens,
        )

        # ── Show which messages were kept ──────────────────────────────
        print("  Retained messages:")
        for m in filtered:
            preview = _text(m["content"])[:55].replace("\n", " ")
            print(f"    [{m['role']:<9}] {preview}...")

        # ── Agent call with filtered context ──────────────────────────
        agent = Agent(
            model=model,
            system_prompt="You are a helpful assistant. Answer concisely (≤ 80 words).",
            messages=filtered,
        )
        response = agent(query)

        if hasattr(response, "message") and response.message:
            content = response.message.get("content", "")
            if isinstance(content, list):
                reply = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            else:
                reply = str(content)
        else:
            reply = str(response)

        print(f"\n  🤖 Agent reply: {reply[:200].replace(chr(10), ' ')}...")

        saved_pct = ((full_tokens - filtered_tokens) / full_tokens * 100) if full_tokens else 0
        viz.print_success(f"Saved {saved_pct:.1f} % of tokens on this call")

    viz.print_kv(
        "Demo 2 Summary",
        {
            "Technique": "Keyword + recency filtering",
            "Recency window": f"{RECENCY_WINDOW} most recent pairs always kept",
            "Key benefit": "Reduces noise and token cost without data loss",
            "Next step": "See DEMO 3 for compression when even SELECT is not enough",
        },
    )


if __name__ == "__main__":
    run_demo()
