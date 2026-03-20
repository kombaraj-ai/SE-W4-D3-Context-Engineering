"""
main_demo.py  –  Run all four Context Engineering demos
=======================================================

Usage
-----
    python main_demo.py              # run all four demos
    python main_demo.py --demo 1     # run only Demo 1 (WRITE)
    python main_demo.py --demo 2,3   # run Demo 2 and 3

Each demo is imported as a module and its ``run_demo()`` function is
called in sequence.  A comparison summary table is printed at the end.
"""

import sys
import os
import argparse
import time
import json

# Make the repo root importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Visualizer

viz = Visualizer()

DEMOS = {
    1: ("WRITE",    "Context Creation & Token Tracking",    "demos.1_context_write"),
    2: ("SELECT",   "Selective Context Passing",             "demos.2_context_select"),
    3: ("COMPRESS", "Context Compression",                   "demos.3_context_compress"),
    4: ("ISOLATE",  "Context Isolation",                     "demos.4_context_isolate"),
}


def run_all(selected: list[int]) -> None:
    results = {}

    for demo_num in selected:
        tag, title, module_path = DEMOS[demo_num]

        viz.print_header(
            f"═══  DEMO {demo_num} · {tag}  ═══",
            title,
        )

        start = time.perf_counter()
        try:
            import importlib
            mod = importlib.import_module(module_path)
            mod.run_demo()
            elapsed = time.perf_counter() - start
            results[demo_num] = {"status": "✓  OK", "elapsed": elapsed}
        except Exception as exc:
            elapsed = time.perf_counter() - start
            results[demo_num] = {"status": f"✗  FAILED: {exc}", "elapsed": elapsed}
            import traceback
            traceback.print_exc()

    # ── Comparison table ───────────────────────────────────────────────
    print()
    viz.print_header("Context Engineering — Run Summary")
    for demo_num, info in results.items():
        tag, title, _ = DEMOS[demo_num]
        print(
            f"  Demo {demo_num} [{tag:<9}]  {title:<40}  "
            f"{info['status']}  ({info['elapsed']:.1f}s)"
        )
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Context Engineering demo runner using AWS Strands Agents"
    )
    p.add_argument(
        "--demo",
        default="1,2,3,4",
        help="Comma-separated demo numbers to run (default: 1,2,3,4)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = [int(x.strip()) for x in args.demo.split(",") if x.strip().isdigit()]
    selected = [d for d in selected if d in DEMOS]

    if not selected:
        print("No valid demo numbers specified. Use --demo 1,2,3,4")
        sys.exit(1)

    run_all(selected)
