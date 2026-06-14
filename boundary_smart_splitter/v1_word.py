from __future__ import annotations

import re
from typing import List, Tuple

from .base import BaseSplitter


class WordSplitter(BaseSplitter):
    """V1: Word-count-based text splitter.

    ``chunk_size`` counts **words** (not characters).  A ``max_chars``
    ceiling ensures no single chunk exceeds embedding-model limits.

    Parameters
    ----------
    chunk_size : int, default 60
        Target number of words per chunk.
    max_chars : int, default 500
        Hard character ceiling — never exceeded.
    tolerance : int, default 10
        Number of words to scan forward or backward when adjusting a
        split boundary that lands mid-word.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 60,
        max_chars: int = 500,
        tolerance: int = 10,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            max_chars=max_chars,
            tolerance=tolerance,
        )

    _WORD_RE = re.compile(r"\S+")

    def _tokenize_words(self, text: str) -> List[Tuple[int, int]]:
        """Return ``[(start, end), ...]`` for each word in *text*.

        A "word" is any run of non-whitespace characters.
        """
        return [m.span() for m in self._WORD_RE.finditer(text)]

    def split(self, text: str) -> List[str]:
        """Split *text* into word-count-respecting chunks.

        ��� Counting words  → group into ``chunk_size`` word groups
        → enforce ``max_chars`` character ceiling
        → extract and strip each chunk.
        """
        if not text:
            return []

        words = self._tokenize_words(text)
        if not words:
            return []

        chunks: List[str] = []
        i = 0
        n_words = len(words)

        while i < n_words:
            start_pos = words[i][0]

            # --- step 1: take up to chunk_size words --------------------
            j = min(i + self.chunk_size, n_words)

            # --- step 2: enforce max_chars (shrink from the right) -----
            while j > i:
                end_pos = words[j - 1][1]
                if (end_pos - start_pos) <= self.max_chars:
                    break
                j -= 1  # drop the last word and try again

            # --- step 2b: single word still too long — hard char cut ---
            if j <= i:
                # Even one word exceeds max_chars — split it at
                # max_chars boundaries (last-resort character split).
                start = words[i][0]
                end = words[i][1]
                while start < end:
                    cut = min(start + self.max_chars, end)
                    piece = text[start:cut].strip()
                    if piece:
                        chunks.append(piece)
                    start = cut
                i += 1
                continue

            # --- step 3: extract the chunk -----------------------------
            end_pos = words[j - 1][1]
            chunk = text[start_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            i = j

        return chunks
