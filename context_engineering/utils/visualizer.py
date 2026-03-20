"""
visualizer.py
-------------
Terminal-friendly visual output helpers for all four demos.

Uses ANSI colour codes via colorama so output is readable on any
platform. Each public method prints a self-contained panel that
can be composed in any order.
"""

import colorama
from colorama import Fore, Back, Style
from typing import List, Dict, Any

colorama.init(autoreset=True)


# ── palette constants ──────────────────────────────────────────────────
C_TITLE   = Fore.CYAN  + Style.BRIGHT
C_OK      = Fore.GREEN + Style.BRIGHT
C_WARN    = Fore.YELLOW
C_ERROR   = Fore.RED   + Style.BRIGHT
C_DIM     = Style.DIM
C_BOLD    = Style.BRIGHT
C_RESET   = Style.RESET_ALL

BAR_FILL  = "█"
BAR_EMPTY = "░"


class Visualizer:
    """
    Collection of static print helpers used across all demo scripts.
    """

    def __init__(self, bar_width: int = 40):
        self.bar_width = bar_width

    # ------------------------------------------------------------------
    # Generic layout primitives
    # ------------------------------------------------------------------

    def print_header(self, title: str, subtitle: str = "") -> None:
        """Print a full-width section header."""
        width = 60
        print()
        print(C_TITLE + "╔" + "═" * (width - 2) + "╗")
        print(C_TITLE + "║" + title.center(width - 2) + "║")
        if subtitle:
            print(C_TITLE + "║" + subtitle.center(width - 2) + "║")
        print(C_TITLE + "╚" + "═" * (width - 2) + "╝")
        print()

    def print_section(self, label: str) -> None:
        print(C_BOLD + f"\n▶ {label}")
        print(C_DIM + "─" * 55)

    def print_success(self, msg: str) -> None:
        print(C_OK + f"  ✓ {msg}")

    def print_info(self, msg: str) -> None:
        print(f"  {msg}")

    def print_warning(self, msg: str) -> None:
        print(C_WARN + f"  ⚠ {msg}")

    # ------------------------------------------------------------------
    # Token bar
    # ------------------------------------------------------------------

    def _make_bar(self, used: int, total: int) -> str:
        """Build a coloured ASCII bar for *used / total* tokens."""
        ratio = min(used / total, 1.0) if total else 0
        filled = int(ratio * self.bar_width)
        empty  = self.bar_width - filled

        colour = C_OK if ratio < 0.6 else (C_WARN if ratio < 0.85 else C_ERROR)
        bar = colour + BAR_FILL * filled + C_DIM + BAR_EMPTY * empty
        pct = f"{ratio * 100:5.1f}%"
        return f"[{bar}{C_RESET}] {pct}  ({used:,} / {total:,} tokens)"

    def print_token_bar(
        self,
        label: str,
        used: int,
        total: int = 8000,
    ) -> None:
        print(f"  {C_BOLD}{label:<22}{C_RESET}  {self._make_bar(used, total)}")

    # ------------------------------------------------------------------
    # Message table
    # ------------------------------------------------------------------

    def print_message_table(
        self, breakdown: List[Dict[str, Any]], context_window: int = 8000
    ) -> None:
        """Print per-message token breakdown as a compact table."""
        self.print_section("Message Token Breakdown")
        header = f"  {'#':>3}  {'Role':<12}  {'Tokens':>7}  {'Cumul.':>7}  Preview"
        print(C_DIM + header)
        print(C_DIM + "  " + "-" * 70)
        for i, row in enumerate(breakdown, 1):
            role_color = Fore.CYAN if row["role"] == "user" else Fore.MAGENTA
            print(
                f"  {i:>3}  "
                f"{role_color}{row['role']:<12}{C_RESET}  "
                f"{row['tokens']:>7,}  "
                f"{row['cumulative']:>7,}  "
                f"{C_DIM}{row['preview']}"
            )
        print()
        self.print_token_bar("Total context", breakdown[-1]["cumulative"] if breakdown else 0, context_window)

    # ------------------------------------------------------------------
    # Compression summary
    # ------------------------------------------------------------------

    def print_compression_stats(self, stats: Dict[str, Any]) -> None:
        """Print before/after compression metrics."""
        self.print_section("Compression Metrics")
        print(f"  Original tokens   : {C_WARN}{stats['original_tokens']:,}{C_RESET}")
        print(f"  Compressed tokens : {C_OK}{stats['compressed_tokens']:,}{C_RESET}")
        print(f"  Tokens saved      : {C_OK}{stats['saved_tokens']:,}{C_RESET}")
        print(f"  Compression ratio : {C_BOLD}{stats['compression_ratio']:.2f}{C_RESET}")
        print(f"  Space saved       : {C_OK}{stats['percentage_saved']:.1f}%{C_RESET}")
        print()
        self.print_token_bar("Before", stats["original_tokens"])
        self.print_token_bar("After ", stats["compressed_tokens"])

    # ------------------------------------------------------------------
    # Isolation table
    # ------------------------------------------------------------------

    def print_isolation_summary(
        self, contexts: List[Dict[str, Any]]
    ) -> None:
        """Print a table of isolated context slots."""
        self.print_section("Isolated Contexts")
        header = f"  {'ID':<20}  {'Messages':>9}  {'Tokens':>8}  Status"
        print(C_DIM + header)
        print(C_DIM + "  " + "-" * 58)
        for ctx in contexts:
            status_col = C_OK if ctx.get("active") else C_DIM
            print(
                f"  {C_BOLD}{ctx['id']:<20}{C_RESET}  "
                f"{ctx['messages']:>9,}  "
                f"{ctx['tokens']:>8,}  "
                f"{status_col}{'active' if ctx.get('active') else 'idle'}{C_RESET}"
            )
        print()

    # ------------------------------------------------------------------
    # Selector summary
    # ------------------------------------------------------------------

    def print_select_summary(
        self,
        total: int,
        selected: int,
        token_before: int,
        token_after: int,
    ) -> None:
        """Show how many messages were filtered and what that saved."""
        self.print_section("Selective Context Results")
        print(f"  Messages before filter : {C_WARN}{total}{C_RESET}")
        print(f"  Messages after  filter : {C_OK}{selected}{C_RESET}")
        print(f"  Messages dropped       : {total - selected}")
        print()
        self.print_token_bar("Before filter", token_before)
        self.print_token_bar("After  filter", token_after)

    # ------------------------------------------------------------------
    # Generic key-value panel
    # ------------------------------------------------------------------

    def print_kv(self, title: str, pairs: Dict[str, str]) -> None:
        self.print_section(title)
        for k, v in pairs.items():
            print(f"  {C_BOLD}{k:<26}{C_RESET}: {v}")
        print()
