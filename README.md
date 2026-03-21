# Week 4 -> Day 2 -> Context Engineering
---

## Context Engineering with AWS Strands Agents Framework

> A hands-on Python project that demonstrates the four fundamental context engineering techniques — **WRITE, SELECT, COMPRESS, ISOLATE** — using the [AWS Strands Agents](https://github.com/strands-agents/sdk-python) framework.

---

### Table of Contents

1. [What Is Context Engineering?](#1-what-is-context-engineering)
2. [Why It Matters?](#2-why-it-matters)
3. [Project Architecture](#3-project-architecture)
4. [Prerequisites](#4-prerequisites)
5. [Setup](#5-setup)
6. [The Four Techniques Explained](#6-the-four-techniques-explained)
   - [WRITE — Context Creation & Token Tracking](#61-write--context-creation--token-tracking)
   - [SELECT — Selective Context Passing](#62-select--selective-context-passing)
   - [COMPRESS — Context Compression](#63-compress--context-compression)
   - [ISOLATE — Context Isolation](#64-isolate--context-isolation)
7. [Code Deep-Dive](#7-code-deep-dive)
   - [utils/token_counter.py](#71-utilstoken_counterpy)
   - [utils/visualizer.py](#72-utilsvisualizerpy)
   - [demos/1_context_write.py](#73-demos1_context_writepy)
   - [demos/2_context_select.py](#74-demos2_context_selectpy)
   - [demos/3_context_compress.py](#75-demos3_context_compresspy)
   - [demos/4_context_isolate.py](#76-demos4_context_isolatepy)
   - [main_demo.py](#77-main_demopy)
8. [Running the Demos](#8-running-the-demos)
9. [Sample Output](#9-sample-output)
10. [AWS Strands Agents — Key Concepts Used](#10-aws-strands-agents--key-concepts-used)
11. [Extending the Project](#11-extending-the-project)
12. [Troubleshooting](#12-troubleshooting)

---

### 1. What Is Context Engineering?

**Context engineering** is the discipline of deliberately crafting, filtering, compressing, and isolating the information that is passed to a Large Language Model (LLM) on every inference call.

When you call an LLM API, you do not just send the latest message — you resend the *entire* conversation history every single time. Each piece of text in that history consumes tokens from the model's **context window** (the maximum amount of text the model can process at once). Context engineering is the set of techniques used to manage this window intelligently.

```
┌────────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW (8 K)                    │
│                                                            │
│  [System Prompt] [Turn 1] [Turn 2] ... [Turn N] [Query]    │
│                                                            │
│  Every token above is paid for. Context engineering        │
│  decides WHICH tokens to include and HOW to represent them.│
└────────────────────────────────────────────────────────────┘
```

---

### 2. Why It Matters

| Without context engineering | With context engineering |
|---|---|
| Token cost grows unboundedly with conversation length | Token cost is bounded or grows slowly |
| Stale/irrelevant turns confuse the model | Only relevant context is included |
| Single context window shared across unrelated tasks | Each task has its own isolated, lean context |
| Long conversations fail once the window is exceeded | Rolling summaries keep the window manageable |

In production systems, context engineering directly affects:

- **Cost** — tokens consumed × price per token
- **Latency** — larger prompts take longer to process
- **Quality** — irrelevant context can degrade answer accuracy
- **Safety** — context leakage between users or tasks can expose private information

---

### 3. Project Architecture

```
context_engineering/
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
├── config.example.json          ← Template — copy to config.json
├── config.json                  ← Your local config (git-ignored)
│
├── utils/
│   ├── __init__.py              ← Re-exports TokenCounter & Visualizer
│   ├── token_counter.py         ← Token counting via tiktoken
│   └── visualizer.py            ← Coloured terminal output helpers
│
├── demos/
│   ├── 1_context_write.py       ← WRITE demo
│   ├── 2_context_select.py      ← SELECT demo
│   ├── 3_context_compress.py    ← COMPRESS demo
│   └── 4_context_isolate.py     ← ISOLATE demo
│
└── main_demo.py                 ← Orchestrator — runs all demos
```

### Dependency map

```
main_demo.py
    └── demos/1..4_*.py
            ├── strands.Agent          (Strands Agents SDK)
            ├── strands.models.BedrockModel
            ├── utils.TokenCounter     ← utils/token_counter.py
            └── utils.Visualizer       ← utils/visualizer.py
```

---

### 4. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | f-strings and type hints used throughout |
| AWS account | any | Bedrock access required |
| AWS credentials | configured | `~/.aws/credentials` or IAM role |
| Bedrock model access | enabled | Enable `amazon.nova-pro-v1` in Bedrock console |

---

### 5. Setup

### 5.1 Clone / extract

```bash
unzip context_engineering.zip
cd context_engineering
```

### 5.2 Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate.bat     # Windows
```

### 5.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 5.4 Configure AWS credentials

```bash
aws configure
# or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
```

### 5.5 Create config.json

```bash
cp config.example.json config.json
# Edit config.json if you want to change the model or region
```

**config.json fields:**

```jsonc
{
  "aws_region": "us-east-1",             // Bedrock region
  "model_id": "us.amazon.nova-pro-v1:0", // Bedrock model ID
  "max_tokens": 2048,                    // Max tokens per completion
  "context_window_limit": 8000,          // Used for bar visualisations
  "demo_settings": {
    "show_token_bars": true,
    "bar_width": 40,                     // Width of the ASCII token bars
    "verbose": false
  }
}
```

---

### 6. The Four Techniques Explained

### 6.1 WRITE — Context Creation & Token Tracking

**Core idea:** Every message you send to an LLM *accumulates* in the context window. This demo makes that growth visible.

```
Turn 1:  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  12.5%  (1,000 / 8,000 tokens)
Turn 4:  ████████████░░░░░░░░░░░░░░░░░░░░  37.5%  (3,000 / 8,000 tokens)
Turn 8:  ████████████████████████░░░░░░░░  75.0%  (6,000 / 8,000 tokens)
```

The WRITE demo does **no** filtering or compression. It is the baseline that motivates the other three techniques.

**Key observation:** Without intervention, context usage grows linearly with conversation length. At some point, the window overflows and you must truncate (losing information) or switch models with a larger window (higher cost).

---

### 6.2 SELECT — Selective Context Passing

**Core idea:** Not every past turn is relevant to the current question. Filtering the history to only relevant messages reduces tokens without losing important information.

```
Full history:  12 messages  →  4,800 tokens  ████████████████████████░░░░░░░░
After SELECT:   4 messages  →  1,600 tokens  ████████░░░░░░░░░░░░░░░░░░░░░░░░
                                             ↑ 67 % tokens saved
```

**Selection strategy used in this demo:**

1. **Recency window** — always keep the last N turns (configurable, default = 2 pairs). Recent context is almost always relevant.
2. **Keyword matching** — extract meaningful words from the current query; keep older turns where those words appear.
3. **System prompt** — always kept (it is not in the message list; Strands handles it separately).

**When SELECT is sufficient:** when topics in a conversation are clearly separable and the current question relates to only a subset of past turns.

**When SELECT is not enough:** when the entire conversation is densely interconnected, or when the conversation is so long that even the relevant subset exceeds the window → use COMPRESS.

---

### 6.3 COMPRESS — Context Compression

**Core idea:** Replace a large number of historical messages with a single compact summary that preserves all key facts and decisions.

```
Before:  12 messages  →  4,800 tokens  ████████████████████████░░░░░░░░
After:    2 messages  →    480 tokens  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                                       ↑ 90 % tokens saved
```

**Compression pattern:**

```python
# 1. Ask a compression agent to summarise the history
summary = compress_with_agent(long_history, model)

# 2. Store the summary as a system-level injection at the top of a new history
compressed = [
    {"role": "user",      "content": f"[SUMMARY]\n{summary}\n[END SUMMARY]"},
    {"role": "assistant", "content": "Understood, I have the context."},
]

# 3. Continue the conversation from this slim baseline
agent = Agent(model=model, messages=compressed)
reply = agent("What should I work on first?")
```

**Trade-off:** Compression is lossy. Verbatim quotes, exact numbers, and fine-grained details may be paraphrased. For most conversational use-cases this is acceptable; for legal or compliance contexts, prefer SELECT or keep the raw history in a separate store.

**Rolling compression:** In production, apply compression automatically when `token_count > threshold`, creating a new summary that includes the previous summary + recent uncompressed turns. This keeps the window bounded indefinitely.

---

### 6.4 ISOLATE — Context Isolation

**Core idea:** Different tasks, users, or roles must never share a context window. Each gets its own hermetically sealed message list.

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ coding_assistant│  │  recipe_advisor │  │  travel_planner │
│                 │  │                 │  │                 │
│ - Python?       │  │ - Omelette tips?│  │ - Japan timing? │
│ - Decorators?   │  │ - Pasta sauce?  │  │ - Kyoto days?   │
│ - GIL?          │  │ - Herbs?        │  │ - Tokyo sights? │
│                 │  │                 │  │                 │
│  NO crossover ──┼──┼── NO crossover ─┼──┼── NO crossover  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Isolation proof:** at the end of the demo, each agent is asked about topics from another context (e.g. the coding agent is asked about the pasta sauce recipe). A correctly isolated agent will say it has no knowledge of it.

**Production use-cases:**

- **Multi-user chat:** each user session gets its own context ID.
- **Agent pipelines:** a planner agent and an executor agent must not share scratch-pad thoughts.
- **Role separation:** a customer-service agent and an internal-analytics agent running in the same process must be isolated.
- **A/B testing:** run two variants of a system prompt in parallel without cross-contamination.

---

### 7. Code Deep-Dive

### 7.1 `utils/token_counter.py`

**Purpose:** Provide consistent, model-agnostic token counts across all demos.

**Why tiktoken?** AWS Bedrock models do not expose a tokenisation endpoint. `tiktoken` with the `cl100k_base` encoding (used by GPT-4) is a well-calibrated proxy — counts are accurate to within ~5 % for Nova-class models.

**Key methods:**

```python
counter = TokenCounter()

# Count a plain string
counter.count_text("Hello, world!")          # → 4

# Count one chat message (includes role + 4 overhead tokens)
counter.count_message({"role": "user", "content": "Hi"})   # → 7

# Count the full message list (includes reply-prime overhead)
counter.count_messages([...])                 # → total tokens

# Get a per-message breakdown with cumulative totals
counter.get_message_breakdown([...])

# Compare original vs compressed history
counter.calculate_compression_ratio(original, compressed)
# → {"original_tokens": 4800, "compressed_tokens": 480, "percentage_saved": 90.0, ...}
```

**Overhead tokens explained:** Chat models wrap each message with role delimiters and special tokens. We add **4 tokens per message** as a conservative estimate of this overhead, plus **3 tokens** at the start of the list as a reply-prime. This matches the formula in OpenAI's official token counting guide.

---

### 7.2 `utils/visualizer.py`

**Purpose:** Provide reusable, coloured terminal output so all four demos look consistent without each demo reimplementing print logic.

**Design decisions:**

- Uses `colorama` (cross-platform ANSI colours) rather than `rich` — smaller dependency, more portable.
- All methods are stateless; each prints immediately and returns `None`.
- Colour coding: green = healthy (<60 %), yellow = caution (60–85 %), red = danger (>85 %).

**Key methods:**

```python
viz = Visualizer(bar_width=40)

viz.print_header("DEMO 1 · WRITE", "Context Creation")
viz.print_section("Turn 3")
viz.print_token_bar("Context", used=3200, total=8000)
viz.print_message_table(breakdown, context_window=8000)
viz.print_compression_stats(stats_dict)
viz.print_isolation_summary(context_list)
viz.print_select_summary(total=12, selected=4, token_before=4800, token_after=1600)
viz.print_kv("Summary", {"key": "value"})
```

---

### 7.3 `demos/1_context_write.py`

**What it does:**

1. Creates a `BedrockModel` and a single `Agent`.
2. Loops through 8 pre-written ML-tutor questions.
3. After each `agent(user_message)` call, reads `agent.messages` to count tokens.
4. Prints a live token bar and warning messages as the window fills.
5. Prints the full per-message breakdown at the end.

**Key code pattern:**

```python
agent = Agent(model=model, system_prompt="You are a concise ML tutor.")

for user_msg in CONVERSATION:
    response = agent(user_msg)                     # Strands appends to agent.messages
    total_tokens = counter.count_messages(agent.messages)
    viz.print_token_bar(f"Context (turn {turn})", total_tokens, CONTEXT_WINDOW)
```

**Why `agent.messages`?** Strands Agents stores the live conversation as `agent.messages` — a list of dicts in standard OpenAI chat format. We read this list directly after each call to get accurate, real-time token counts.

---

### 7.4 `demos/2_context_select.py`

**What it does:**

1. Pre-populates a `SEED_HISTORY` of 12 messages spanning three topics (Python, ML, French cuisine).
2. For three different queries, calls `select_relevant_messages()` to filter the history.
3. Builds a fresh `Agent` per query, pre-seeded with only the filtered messages.
4. Prints before/after token counts and the set of retained messages.

**Key function — `select_relevant_messages`:**

```python
def select_relevant_messages(history, query, recency=2):
    keywords = extract_keywords(query)     # ["pytorch", "tensorflow", "deep", "learning"]
    
    # Group into (user, assistant) pairs
    pairs = group_into_pairs(history)
    
    recent_pairs = pairs[-recency:]        # Always keep last 2 pairs
    older_pairs  = pairs[:-recency]
    
    selected = []
    for user_msg, asst_msg in older_pairs:
        combined = (user_msg["content"] + asst_msg["content"]).lower()
        if any(kw in combined for kw in keywords):
            selected.extend([user_msg, asst_msg])
    
    for pair in recent_pairs:
        selected.extend(pair)
    
    return selected
```

**Why a fresh Agent per call?** Strands does not currently expose a pre-call hook to mutate its internal message list. Creating a fresh Agent with `messages=filtered_list` is the clean, idiomatic way to inject a custom history. This is a well-known pattern in agentic frameworks — stateless inference with explicit state injection.

---

### 7.5 `demos/3_context_compress.py`

**What it does:**

1. Defines a `LONG_HISTORY` of 12 messages about building a web application.
2. Shows the baseline token cost.
3. Calls a *compression agent* — a separate `Agent` instance whose sole purpose is to summarise.
4. The compression agent is equipped with a `@tool` decorator function `summarise_conversation` that formats the prompt correctly.
5. Wraps the returned summary in a 2-message compressed history.
6. Prints compression statistics.
7. Runs a follow-up question against the compressed context to prove quality is maintained.

**The `@tool` decorator:**

```python
from strands import tool

@tool
def summarise_conversation(messages_json: str) -> str:
    """
    Summarise a JSON-encoded list of chat messages into a compact paragraph.
    ...
    """
    return (
        "Please produce a ≤150-word summary ... \n\n"
        f"Messages:\n{messages_json}"
    )
```

The `@tool` decorator registers the Python function as a callable tool that the Strands Agent can invoke. The docstring becomes the tool description that the LLM uses to decide when to call it. The function parameters become the tool's input schema.

**Compressed history structure:**

```python
[
    {
        "role": "user",
        "content": "[CONVERSATION SUMMARY]\n<summary text>\n[END SUMMARY]"
    },
    {
        "role": "assistant",
        "content": "Understood. I have the context and am ready to continue."
    }
]
```

This is the minimal viable compressed context. The summary is injected as a user turn (not a system message) because Strands passes system prompts separately and we want the summary to appear in the message history.

---

### 7.6 `demos/4_context_isolate.py`

**What it does:**

1. Defines a `ContextManager` class that holds a dict of `context_id → {system_prompt, messages}`.
2. Creates three context slots: `coding_assistant`, `recipe_advisor`, `travel_planner`.
3. Runs three independent multi-turn conversations, one per slot.
4. Prints a table showing messages and token usage per slot.
5. Runs **isolation probes** — asks each agent about a topic from another context.

**`ContextManager.chat()` — the core isolation method:**

```python
def chat(self, context_id: str, user_message: str) -> str:
    ctx = self._contexts[context_id]
    
    # Fresh agent seeded with ONLY this context's history
    agent = Agent(
        model=self.model,
        system_prompt=ctx["system_prompt"],
        messages=list(ctx["messages"]),    # copy — never mutate the stored list
    )
    
    self._append(context_id, "user", user_message)    # record user turn
    response = agent(user_message)
    reply = extract_text(response)
    self._append(context_id, "assistant", reply)      # record assistant turn
    
    return reply
```

The isolation guarantee comes from the fact that `messages=list(ctx["messages"])` is a context-specific list. The agent for `coding_assistant` never sees the messages from `recipe_advisor` and vice versa.

---

### 7.7 `main_demo.py`

**What it does:** Provides a CLI entry point that imports and runs any combination of the four demos in sequence.

```bash
python main_demo.py              # run all four
python main_demo.py --demo 1     # run only WRITE
python main_demo.py --demo 2,3   # run SELECT and COMPRESS
```

**Implementation:** Uses `importlib.import_module()` to dynamically import each demo module by dotted path, then calls its `run_demo()` function. Elapsed time is measured per demo and printed in a final summary table.

---

### 8. Running the Demos

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all four demos (recommended first run)
python main_demo.py

# Run a single demo
python demos/1_context_write.py
python demos/2_context_select.py
python demos/3_context_compress.py
python demos/4_context_isolate.py

# Run specific demos via main
python main_demo.py --demo 3,4
```

**Expected runtime:** 2–5 minutes for all four demos (each demo makes 3–10 Bedrock API calls).

---

### 9. Sample Output

```
╔══════════════════════════════════════════════════════════╗
║                   DEMO 1 · WRITE                         ║
║           Context Creation and Token Tracking            ║
╚══════════════════════════════════════════════════════════╝

  Context window limit: 8,000 tokens

▶ Turn 1 / 8
────────────────────────────────────────────────────────
  🧑 USER : Hi! Can you explain what machine learning is...
  🤖 AGENT: Machine learning is a subset of AI that enab...
  Context (turn 1)        [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  11.2%  (896 / 8,000 tokens)

▶ Turn 4 / 8
────────────────────────────────────────────────────────
  🧑 USER : Give me a quick overview of neural networks...
  🤖 AGENT: Neural networks consist of layers of interco...
  Context (turn 4)        [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  38.4%  (3,072 / 8,000 tokens)
  ⚠  Context > 60 % — approaching compression threshold.
```

---

### 10. AWS Strands Agents — Key Concepts Used

| Concept | How it is used in this project |
|---|---|
| `Agent` class | Central abstraction; wraps model + system prompt + message history |
| `BedrockModel` | Connects to Amazon Bedrock; configured via `aws_region` + `model_id` |
| `agent.messages` | Accessed directly to read live conversation state for token counting |
| `messages=` constructor arg | Used to pre-seed agents with custom/filtered/compressed histories |
| `@tool` decorator | Registers Python functions as LLM-callable tools (Demo 3) |
| Stateless invocation | Each `agent(user_message)` call returns a `AgentResult`; text extracted from `response.message["content"]` |

**Strands Agents message format** follows the standard chat completion format:

```python
[
    {"role": "user",      "content": "What is X?"},
    {"role": "assistant", "content": "X is ..."},
    {"role": "user",      "content": "Tell me more."},
]
```

When you pass `messages=[...]` to the `Agent` constructor, Strands uses that list as the conversation history for the next call, then appends the new user + assistant turns to it internally.

---

### 11. Extending the Project

### Add a new context engineering technique

1. Create `demos/5_context_your_technique.py`.
2. Implement a `run_demo()` function.
3. Add an entry to the `DEMOS` dict in `main_demo.py`.

### Implement rolling compression (production pattern)

```python
COMPRESS_THRESHOLD = 0.75   # compress when context > 75 % full

def maybe_compress(agent, model, counter, context_window):
    tokens = counter.count_messages(agent.messages)
    if tokens / context_window > COMPRESS_THRESHOLD:
        summary = compress_with_agent(agent.messages, model)
        agent.messages = build_compressed_history(summary)
```

### Swap the model

Edit `config.json`:

```json
{
  "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
  "aws_region": "us-east-1"
}
```

Ensure the model is enabled in your Bedrock console.

### Add vector-search based SELECT

Replace keyword matching in `select_relevant_messages` with cosine similarity between sentence embeddings (e.g. using `sentence-transformers`) for more accurate relevance filtering.

---

### 12. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `NoCredentialsError` | AWS credentials not configured | Run `aws configure` or set `AWS_*` env vars |
| `AccessDeniedException` | Model not enabled in Bedrock | Enable `amazon.nova-pro-v1` in Bedrock → Model access |
| `FileNotFoundError: config.json` | Config file missing | `cp config.example.json config.json` |
| `ModuleNotFoundError: strands` | Dependencies not installed | `pip install -r requirements.txt` |
| `ValidationException` from Bedrock | Model ID wrong for region | Check model availability in your region |
| Slow responses | First call warms up Bedrock endpoint | Normal — subsequent calls are faster |

---

### Licence

MIT — free to use, modify, and distribute with attribution.

---

*Built with [AWS Strands Agents](https://github.com/strands-agents/sdk-python) · Visualised with [colorama](https://pypi.org/project/colorama/) · Token counting via [tiktoken](https://pypi.org/project/tiktoken/)*
