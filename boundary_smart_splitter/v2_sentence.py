from __future__ import annotations

from typing import Callable, Generator, List, Optional, Tuple, Union

from .base import BaseSplitter


class SentenceSplitter(BaseSplitter):
    """V2: Sentence-count-based text splitter.

    ``chunk_size`` counts **sentences** (not characters).  When a single
    sentence exceeds ``max_chars`` (or ``max_tokens``) the splitter falls back to V1
    (word-boundary) behaviour within that sentence.

    Instances of this class are thread-safe. Each ``split()`` call is stateless.

    Parameters
    ----------
    chunk_size : int, default 5
        Target number of sentences per chunk.
    max_chars : int, optional, default 500
        Hard character ceiling.
    max_tokens : int, optional
        Hard token ceiling.
    tolerance : int, default 2
        Number of sentences to scan forward/backward when adjusting
        a boundary.
    boundary_preference : str, default "forward"
        ``"forward"`` or ``"backward"``.
    abbreviations : set[str] | None, optional
        Abbreviations whose periods should not be treated as sentence endings.
    length_function : str | Callable[[str], int], default "cl100k_base"
        The encoding name, model name, or callable used to calculate token count.
    """

    DEFAULT_ABBREVIATIONS = {
        "e.g.",
        "i.e.",
        "etc.",
        "Dr.",
        "Mr.",
        "Mrs.",
        "Ms.",
        "Prof.",
        "Sr.",
        "Jr.",
        "vs.",
        "approx.",
        "U.S.A.",
        "Jan.",
        "Feb.",
        "Mar.",
        "Apr.",
        "Jun.",
        "Jul.",
        "Aug.",
        "Sep.",
        "Sept.",
        "Oct.",
        "Nov.",
        "Dec.",
    }

    def __init__(
        self,
        *,
        chunk_size: int = 5,
        max_chars: Optional[int] = 500,
        max_tokens: Optional[int] = None,
        tolerance: int = 2,
        boundary_preference: str = "forward",
        abbreviations: set[str] | None = None,
        length_function: Union[str, Callable[[str], int]] = "cl100k_base",
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            max_chars=max_chars,
            max_tokens=max_tokens,
            tolerance=tolerance,
            length_function=length_function,
        )
        if boundary_preference not in ("forward", "backward"):
            raise ValueError('boundary_preference must be "forward" or "backward"')
        self.boundary_preference = boundary_preference
        self.abbreviations = frozenset(
            abbreviations if abbreviations is not None
            else self.DEFAULT_ABBREVIATIONS
        )

        # Precompute abbreviation metadata for fast period detection.
        abbr_offsets: list[tuple[str, int, int]] = []
        for abbr in self.abbreviations:
            lower = abbr.lower()
            for i, ch in enumerate(lower):
                if ch == '.':
                    abbr_offsets.append((lower, i, len(lower)))
        self._abbr_offsets = tuple(abbr_offsets)

        # Lazy V1 fallback splitter.
        self._v1_splitter: Optional["WordSplitter"] = None

    @property
    def _word_splitter(self) -> "WordSplitter":
        if self._v1_splitter is None:
            from .v1_word import WordSplitter

            self._v1_splitter = WordSplitter(
                chunk_size=self.chunk_size * 5,  # generous word budget per sentence
                max_chars=self.max_chars,
                max_tokens=self.max_tokens,
                tolerance=self.tolerance * 5,
                length_function=self.length_function,
            )
        return self._v1_splitter

    # ── sentence tokenisation ───────────────────────────────────────

    def _is_abbreviation(self, text: str, pos: int) -> bool:
        """Return True if the period at *pos* is part of a known abbreviation."""
        if pos < 0 or pos >= len(text) or text[pos] != ".":
            return False
        for abbr, period_offset, abbr_len in self._abbr_offsets:
            start = pos - period_offset
            if start >= 0 and text[start:start + abbr_len].lower() == abbr:
                return True
        return False

    def _tokenize_sentences(self, text: str) -> List[Tuple[int, int]]:
        """Return ``[(start, end), ...]`` for each sentence in *text*.

        Sentence ends at ``.?!`` unless the period is part of a known
        abbreviation.  Trailing whitespace is *included* in the end
        position so that the next sentence starts at the first
        non-whitespace character of its first word.
        """
        if not text:
            return []

        sentences: List[Tuple[int, int]] = []
        sent_start = 0
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            if ch in ".?!" and not self._is_abbreviation(text, i):
                # End of sentence — include the punctuation and any
                # trailing whitespace before the next sentence.
                end = i + 1
                # Advance through trailing whitespace.
                while end < n and text[end].isspace():
                    end += 1
                sentences.append((sent_start, end))
                sent_start = end
                i = end
                continue
            i += 1

        # Last "sentence" (may be empty if text ends with punctuation).
        if sent_start < n:
            sentences.append((sent_start, n))

        return sentences

    # ── splitting logic ─────────────────────────────────────────────

    def split_stream(self, text: str) -> Generator[str, None, None]:
        """Yield chunks of *text* as they are formed."""
        if not text:
            return

        sentences = self._tokenize_sentences(text)
        if not sentences:
            # No sentence boundaries found — fall back to V1.
            yield from self._word_splitter.split_stream(text)
            return

        i = 0
        n = len(sentences)

        while i < n:
            sent_start, _ = sentences[i]

            # --- step 1: take up to chunk_size sentences ------------
            j = min(i + self.chunk_size, n)

            # --- step 2: enforce max_chars and max_tokens by dropping sentences ----
            while j > i:
                _, last_end = sentences[j - 1]
                substring = text[sent_start:last_end]
                
                # Check character limit
                char_ok = self.max_chars is None or (last_end - sent_start) <= self.max_chars
                # Check token limit
                token_ok = self.max_tokens is None or self._len_fn(substring) <= self.max_tokens
                
                if char_ok and token_ok:
                    break
                j -= 1

            # --- step 2b: single sentence still too long — V1 fallback
            if j <= i:
                s_start = sentences[i][0]
                s_end = sentences[i][1]
                s_text = text[s_start:s_end].strip()
                if s_text:
                    yield from self._word_splitter.split_stream(s_text)
                i += 1
                continue

            # --- step 3: extract the chunk ---------------------------
            _, last_end = sentences[j - 1]
            chunk = text[sent_start:last_end].strip()
            if chunk:
                yield chunk

            i = j

    def split(self, text: str) -> List[str]:
        """Split *text* into sentence-count-respecting chunks."""
        return list(self.split_stream(text))
