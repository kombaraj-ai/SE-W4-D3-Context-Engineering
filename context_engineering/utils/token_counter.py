"""
token_counter.py
----------------
Utility for counting tokens in messages and conversations.

Uses tiktoken (OpenAI's BPE tokenizer) as a close proxy for
token counts in most modern LLMs including Amazon Bedrock models.
The counts are approximate but consistent enough for demonstration
purposes.
"""

import tiktoken
from typing import List, Dict, Any, Union


class TokenCounter:
    """
    Wraps tiktoken to provide token counting for message lists.

    Attributes
    ----------
    encoding : tiktoken.Encoding
        The BPE encoding used for counting. We use 'cl100k_base',
        the encoding used by GPT-4 / Claude-class models.
    """

    def __init__(self, model: str = "cl100k_base"):
        """
        Parameters
        ----------
        model : str
            Name of the tiktoken encoding to load.
        """
        self.encoding = tiktoken.get_encoding(model)

    # ------------------------------------------------------------------
    # Content normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(content: Union[str, list, None]) -> str:
        """
        Normalise a message's 'content' field to a plain string.

        Strands Agents stores content as a list of typed blocks, e.g.:
            [{"type": "text", "text": "Hello, world!"}]

        Plain strings (the standard OpenAI format) are returned as-is.
        None / unknown types fall back to an empty string.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    # Strands "text" block: {"type": "text", "text": "..."}
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    # Fallback: any dict with a plain "text" key
                    elif "text" in block:
                        parts.append(str(block["text"]))
                    # Fallback: any dict with a plain "content" key
                    elif "content" in block:
                        parts.append(str(block["content"]))
                else:
                    parts.append(str(block))
            return " ".join(parts)
        # Catch-all for any other type
        return str(content)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def count_text(self, text: str) -> int:
        """Return the number of tokens in a plain text string."""
        return len(self.encoding.encode(text))

    def count_message(self, message: Dict[str, Any]) -> int:
        """
        Count tokens for a single chat message dict.

        Handles both plain-string content (OpenAI format) and
        content-block lists (Strands Agents format).

        Adds 4 overhead tokens per message to approximate the
        role/delimiter framing that chat models consume.
        """
        text = self._extract_text(message.get("content"))
        base = self.count_text(text)
        base += self.count_text(message.get("role", ""))
        return base + 4  # message framing overhead

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count total tokens for a list of chat messages.

        Adds 3 tokens for the reply-prime token used by chat models.
        """
        total = sum(self.count_message(m) for m in messages)
        return total + 3  # reply-prime overhead

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_message_breakdown(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Return per-message token counts alongside cumulative totals.

        Returns
        -------
        list of dict
            Each dict has keys: role, preview, tokens, cumulative.
        """
        breakdown = []
        cumulative = 0
        for msg in messages:
            t = self.count_message(msg)
            cumulative += t
            text = self._extract_text(msg.get("content"))
            breakdown.append(
                {
                    "role": msg.get("role", "unknown"),
                    "preview": text[:60] + "...",
                    "tokens": t,
                    "cumulative": cumulative,
                }
            )
        return breakdown

    def calculate_compression_ratio(
        self,
        original: List[Dict[str, str]],
        compressed: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Compare original vs. compressed message lists.

        Returns
        -------
        dict
            original_tokens, compressed_tokens, saved_tokens,
            compression_ratio (0–1 float), percentage_saved (float).
        """
        orig_tokens = self.count_messages(original)
        comp_tokens = self.count_messages(compressed)
        saved = orig_tokens - comp_tokens
        ratio = comp_tokens / orig_tokens if orig_tokens else 0
        pct = (1 - ratio) * 100

        return {
            "original_tokens": orig_tokens,
            "compressed_tokens": comp_tokens,
            "saved_tokens": saved,
            "compression_ratio": ratio,
            "percentage_saved": pct,
        }
