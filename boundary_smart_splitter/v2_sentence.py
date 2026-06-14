from __future__ import annotations

from typing import List, Optional, Tuple

from .base import BaseSplitter


class SentenceSplitter(BaseSplitter):
    """V2: Sentence-count-based text splitter.

    ``chunk_size`` counts **sentences** (not characters).  When a single
    sentence exceeds ``max_chars`` the splitter falls back to V1
    (word-boundary) behaviour within that sentence.

    Parameters
    ----------
    chunk_size : int, default 5
        Target number of sentences per chunk.
    max_chars : int, default 500
        Hard character ceiling — if a chunk of ``chunk_size`` sentences
        exceeds this, the sentence group is passed to V1 for
        word-level splitting.
    tolerance : int, default 2
        Number of sentences to scan forward/backward when adjusting
        a boundary.
    boundary_preference : str, default "forward"
        ``"forward"`` (prefer a longer chunk, extending forward to the
        next sentence end) or ``"backward"`` (prefer a shorter chunk,
        going back to the previous sentence end).
    abbreviations : set[str] | None, optional
        Abbreviations whose periods should not be treated as sentence
        endings.  Defaults to a standard set of English abbreviations.
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
        max_chars: int = 500,
        tolerance: int = 2,
        boundary_preference: str = "forward",
        abbreviations: set[str] | None = None,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            max_chars=max_chars,
            tolerance=tolerance,
        )
        if boundary_preference not in ("forward", "backward"):
            raise ValueError('boundary_preference must be "forward" or "backward"')
        self.boundary_preference = boundary_preference
        self.abbreviations = (
            abbreviations if abbreviations is not None
            else self.DEFAULT_ABBREVIATIONS.copy()
        )

        # Precompute abbreviation metadata for fast period detection.
        self._abbr_offsets: list[tuple[str, int, int]] = []
        for abbr in self.abbreviations:
            lower = abbr.lower()
            for i, ch in enumerate(lower):
                if ch == '.':
                    self._abbr_offsets.append((lower, i, len(lower)))

        # Lazy V1 fallback splitter.
        self._v1_splitter: Optional["WordSplitter"] = None

    @property
    def _word_splitter(self) -> "WordSplitter":
        if self._v1_splitter is None:
            from .v1_word import WordSplitter

            self._v1_splitter = WordSplitter(
                chunk_size=self.chunk_size * 5,  # generous word budget per sentence
                max_chars=self.max_chars,
                tolerance=self.tolerance * 5,
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

    def split(self, text: str) -> List[str]:
        """Split *text* into sentence-count-respecting chunks."""
        if not text:
            return []

        sentences = self._tokenize_sentences(text)
        if not sentences:
            # No sentence boundaries found — fall back to V1.
            return self._word_splitter.split(text)

        chunks: List[str] = []
        i = 0
        n = len(sentences)

        while i < n:
            sent_start, _ = sentences[i]

            # --- step 1: take up to chunk_size sentences ------------
            j = min(i + self.chunk_size, n)
            _, sent_end = sentences[j - 1]
            chunk_len = sent_end - sent_start

            # --- step 2: enforce max_chars by dropping sentences ----
            while j > i:
                _, last_end = sentences[j - 1]
                if (last_end - sent_start) <= self.max_chars:
                    break
                j -= 1

            # --- step 2b: single sentence still too long — V1 fallback
            if j <= i:
                # The current sentence alone exceeds max_chars.
                # Fall back to V1 word splitting for this sentence.
                s_start = sentences[i][0]
                s_end = sentences[i][1]
                s_text = text[s_start:s_end].strip()
                if s_text:
                    v1_parts = self._word_splitter.split(s_text)
                    chunks.extend(v1_parts)
                i += 1
                continue

            # --- step 3: extract the chunk ---------------------------
            _, last_end = sentences[j - 1]
            chunk = text[sent_start:last_end].strip()
            if chunk:
                chunks.append(chunk)

            i = j

        return chunks
