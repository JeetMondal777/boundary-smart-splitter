from __future__ import annotations

import re
from typing import Callable, Generator, List, Optional, Tuple, Union

from .base import BaseSplitter


class WordSplitter(BaseSplitter):
    """V1: Word-count-based text splitter.

    ``chunk_size`` counts **words** (not characters).  A ``max_chars``
    ceiling ensures no single chunk exceeds embedding-model limits.

    Instances of this class are thread-safe. Each ``split()`` call is stateless.

    Parameters
    ----------
    chunk_size : int, default 60
        Target number of words per chunk.
    max_chars : int, optional, default 500
        Hard character ceiling — never exceeded.
    max_tokens : int, optional
        Hard token ceiling — never exceeded.
    tolerance : int, default 10
        Number of words to scan forward or backward when adjusting a
        split boundary that lands mid-word.
    length_function : str | Callable[[str], int], default "cl100k_base"
        The encoding name, model name, or callable used to calculate token count.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 60,
        max_chars: Optional[int] = 500,
        max_tokens: Optional[int] = None,
        tolerance: int = 10,
        length_function: Union[str, Callable[[str], int]] = "cl100k_base",
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            max_chars=max_chars,
            max_tokens=max_tokens,
            tolerance=tolerance,
            length_function=length_function,
        )

    _WORD_RE = re.compile(r"\S+")

    def _tokenize_words(self, text: str) -> List[Tuple[int, int]]:
        """Return ``[(start, end), ...]`` for each word in *text*.

        A "word" is any run of non-whitespace characters.
        """
        return [m.span() for m in self._WORD_RE.finditer(text)]

    def split_stream(self, text: str) -> Generator[str, None, None]:
        """Yield chunks of *text* as they are formed."""
        if not text:
            return

        words = self._tokenize_words(text)
        if not words:
            return

        i = 0
        n_words = len(words)

        while i < n_words:
            start_pos = words[i][0]

            # --- step 1: take up to chunk_size words --------------------
            j = min(i + self.chunk_size, n_words)

            # --- step 2: enforce max_chars and max_tokens bounds ------
            while j > i:
                end_pos = words[j - 1][1]
                substring = text[start_pos:end_pos]
                
                # Check character limit
                char_ok = self.max_chars is None or (end_pos - start_pos) <= self.max_chars
                # Check token limit
                token_ok = self.max_tokens is None or self._len_fn(substring) <= self.max_tokens
                
                if char_ok and token_ok:
                    break
                j -= 1  # drop the last word and try again

            # --- step 2b: single word still too long — hard cut ---
            if j <= i:
                # Even one word exceeds limits — split it at
                # maximum allowable boundaries.
                start = words[i][0]
                end = words[i][1]
                while start < end:
                    cut = self._find_cut_point(text, start, end)
                    piece = text[start:cut].strip()
                    if piece:
                        yield piece
                    start = cut
                i += 1
                continue

            # --- step 3: extract the chunk -----------------------------
            end_pos = words[j - 1][1]
            chunk = text[start_pos:end_pos].strip()
            if chunk:
                yield chunk

            i = j

    def split(self, text: str) -> List[str]:
        """Split *text* into word-count-respecting chunks."""
        return list(self.split_stream(text))
