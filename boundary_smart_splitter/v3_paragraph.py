from __future__ import annotations

import re
from typing import Callable, Generator, List, Optional, Tuple, Union

from .base import BaseSplitter


class ParagraphSplitter(BaseSplitter):
    """V3: Paragraph-count-based text splitter.

    ``chunk_size`` counts **paragraphs** (default 2).  When a single
    paragraph exceeds ``max_chars`` (or ``max_tokens``) the splitter falls back to V2
    (sentence-boundary) behaviour within that paragraph.

    Instances of this class are thread-safe. Each ``split()`` call is stateless.

    Parameters
    ----------
    chunk_size : int, default 2
        Target number of paragraphs per chunk.
    max_chars : int, optional, default 500
        Hard character ceiling — if a chunk of ``chunk_size`` paragraphs
        exceeds this, the paragraph group is passed to V2 for
        sentence-level splitting.
    max_tokens : int, optional
        Hard token ceiling.
    tolerance : int, default 1
        Number of paragraphs to scan forward/backward when adjusting
        a boundary.
    overlap : int, default 0
        Number of trailing **characters** of each chunk to repeat at
        the start of the next chunk for context continuity.
    paragraph_separator : str, default ``"\\n\\n"``
        String that separates paragraphs.
    use_markdown_mode : bool, default False
        When True, also treats Markdown horizontal rules (``---``,
        ``***``) and headings (``#``, ``##``) as paragraph boundaries.
    boundary_preference : str, default "forward"
        Passed through to V2 SentenceSplitter when falling back.
    abbreviations : set[str] | None, optional
        Passed through to V2 SentenceSplitter.
    length_function : str | Callable[[str], int], default "cl100k_base"
        The encoding name, model name, or callable used to calculate token count.
    """

    _MD_HR_RE = re.compile(r"\n[-*_]{3,}\n")
    _MD_HEADING_RE = re.compile(r"\n#{1,6} [^\n]+\n")

    def __init__(
        self,
        *,
        chunk_size: int = 2,
        max_chars: Optional[int] = 500,
        max_tokens: Optional[int] = None,
        tolerance: int = 1,
        overlap: int = 0,
        paragraph_separator: str = "\n\n",
        use_markdown_mode: bool = False,
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
        if overlap < 0:
            raise ValueError("overlap must be a non-negative integer")
        self.overlap = overlap
        self.paragraph_separator = paragraph_separator
        self.use_markdown_mode = use_markdown_mode
        self._boundary_preference = boundary_preference
        self._abbreviations = abbreviations

        self._v2_splitter: Optional["SentenceSplitter"] = None

    @property
    def _sentence_splitter(self) -> "SentenceSplitter":
        if self._v2_splitter is None:
            from .v2_sentence import SentenceSplitter

            self._v2_splitter = SentenceSplitter(
                chunk_size=self.chunk_size * 3,  # generous sentence budget
                max_chars=self.max_chars,
                max_tokens=self.max_tokens,
                tolerance=self.tolerance * 2,
                boundary_preference=self._boundary_preference,
                abbreviations=self._abbreviations,
                length_function=self.length_function,
            )
        return self._v2_splitter

    def _tokenize_paragraphs(self, text: str) -> List[Tuple[int, int]]:
        """Return ``[(start, end), ...]`` for each paragraph in *text*.

        Paragraphs are delimited by ``paragraph_separator`` (default
        ``\\n\\n``).  The separator itself is consumed; each paragraph
        includes its trailing newline if any.
        """
        if not text:
            return []

        paragraphs: List[Tuple[int, int]] = []
        start = 0
        sep = self.paragraph_separator
        pos = 0

        while True:
            idx = text.find(sep, pos)
            if idx == -1:
                # No more separators — rest of text is one paragraph.
                paragraphs.append((start, len(text)))
                break

            # Paragraph ends at the separator (exclusive).
            paragraphs.append((start, idx))
            # Next paragraph starts right after the separator.
            start = idx + len(sep)
            pos = idx + len(sep)
            # If the separator appears at the very end, we have an empty paragraph.
            if pos >= len(text):
                paragraphs.append((start, len(text)))
                break

        if self.use_markdown_mode:
            # Overlay Markdown-only boundaries onto the plain paragraphs.
            md_boundaries: List[int] = []
            for m in self._MD_HR_RE.finditer(text):
                md_boundaries.append(m.end())
            for m in self._MD_HEADING_RE.finditer(text):
                md_boundaries.append(m.start())
            md_boundaries = sorted(set(md_boundaries))

            # Re-split paragraphs at Markdown boundaries.
            if md_boundaries:
                paragraphs = self._re_split_with_boundaries(text, paragraphs, md_boundaries)

        return paragraphs

    def _re_split_with_boundaries(
        self,
        text: str,
        orig_paras: List[Tuple[int, int]],
        boundaries: List[int],
    ) -> List[Tuple[int, int]]:
        """Re-split existing :attr:`orig_paras` at every position in *boundaries*."""
        new_paras: List[Tuple[int, int]] = []
        for p_start, p_end in orig_paras:
            seg_start = p_start
            for b in boundaries:
                if seg_start < b <= p_end:
                    new_paras.append((seg_start, b))
                    seg_start = b
            if seg_start < p_end:
                new_paras.append((seg_start, p_end))
        return new_paras

    def split_stream(self, text: str) -> Generator[str, None, None]:
        """Yield chunks of *text* as they are formed."""
        if not text:
            return

        paragraphs = self._tokenize_paragraphs(text)
        if not paragraphs:
            return

        carry_overlap: str = ""
        n = len(paragraphs)
        i = 0

        while i < n:
            p_start, _ = paragraphs[i]

            # --- step 1: take up to chunk_size paragraphs -------------
            j = min(i + self.chunk_size, n)

            # --- step 2: enforce max_chars and max_tokens (drop paragraphs) ---------
            while j > i:
                _, last_end = paragraphs[j - 1]
                substring = text[p_start:last_end]
                
                # Check character limit
                char_ok = self.max_chars is None or (last_end - p_start) <= self.max_chars
                # Check token limit
                token_ok = self.max_tokens is None or self._len_fn(substring) <= self.max_tokens
                
                if char_ok and token_ok:
                    break
                j -= 1

            # --- step 2b: single paragraph too long → V2 fallback ----
            if j <= i:
                p_start_i, p_end_i = paragraphs[i]
                p_text = text[p_start_i:p_end_i].strip()
                if p_text:
                    v2_parts = self._sentence_splitter.split(p_text)

                    # Apply carry_overlap to the first V2 piece.
                    if v2_parts:
                        yield carry_overlap + v2_parts[0]
                        prev = v2_parts[0]
                        for vp in v2_parts[1:]:
                            carry = prev[-self.overlap:] if self.overlap > 0 else ""
                            yield carry + vp
                            prev = vp
                        # After the last V2 sub-chunk, set carry_overlap for next.
                        if self.overlap > 0 and prev:
                            carry_overlap = prev[-self.overlap:]
                    carry_overlap = ""
                i += 1
                continue

            # --- step 3: extract the paragraph group ------------------
            _, last_end = paragraphs[j - 1]
            raw = text[p_start:last_end].strip()
            chunk = carry_overlap + raw if carry_overlap else raw
            if chunk.strip():
                yield chunk

            # Carry overlap for the next paragraph chunk.
            if self.overlap > 0 and raw:
                carry_overlap = raw[-self.overlap:]

            i = j

        if carry_overlap.strip():
            yield carry_overlap

    def split(self, text: str) -> List[str]:
        """Split *text* into paragraph-count-respecting chunks."""
        return list(self.split_stream(text))
