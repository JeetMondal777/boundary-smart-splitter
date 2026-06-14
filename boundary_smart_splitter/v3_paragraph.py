from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .base import BaseSplitter


class ParagraphSplitter(BaseSplitter):
    """V3: Paragraph-count-based text splitter.

    ``chunk_size`` counts **paragraphs** (default 2).  When a single
    paragraph exceeds ``max_chars`` the splitter falls back to V2
    (sentence-boundary) behaviour within that paragraph.

    Parameters
    ----------
    chunk_size : int, default 2
        Target number of paragraphs per chunk.
    max_chars : int, default 500
        Hard character ceiling — if a chunk of ``chunk_size`` paragraphs
        exceeds this, the paragraph group is passed to V2 for
        sentence-level splitting.
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
    """

    _MD_HR_RE = re.compile(r"\n[-*_]{3,}\n")
    _MD_HEADING_RE = re.compile(r"\n#{1,6} [^\n]+\n")

    def __init__(
        self,
        *,
        chunk_size: int = 2,
        max_chars: int = 500,
        tolerance: int = 1,
        overlap: int = 0,
        paragraph_separator: str = "\n\n",
        use_markdown_mode: bool = False,
        boundary_preference: str = "forward",
        abbreviations: set[str] | None = None,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            max_chars=max_chars,
            tolerance=tolerance,
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
                tolerance=self.tolerance * 2,
                boundary_preference=self._boundary_preference,
                abbreviations=self._abbreviations,
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

    def split(self, text: str) -> List[str]:
        """Split *text* into paragraph-count-respecting chunks."""
        if not text:
            return []

        paragraphs = self._tokenize_paragraphs(text)
        if not paragraphs:
            return []

        chunks: List[str] = []
        carry_overlap: str = ""
        n = len(paragraphs)
        i = 0

        while i < n:
            p_start, _ = paragraphs[i]

            # --- step 1: take up to chunk_size paragraphs -------------
            j = min(i + self.chunk_size, n)
            _, p_end = paragraphs[j - 1]
            chunk_len = p_end - p_start

            # --- step 2: enforce max_chars (drop paragraphs) ---------
            while j > i:
                _, last_end = paragraphs[j - 1]
                if (last_end - p_start) <= self.max_chars:
                    break
                j -= 1

            # --- step 2b: single paragraph too long → V2 fallback ----
            if j <= i:
                p_start_i, p_end_i = paragraphs[i]
                p_text = text[p_start_i:p_end_i].strip()
                if p_text:
                    v2_parts = self._sentence_splitter.split(p_text)

                    # Apply carry_overlap to the first V2 piece.
                    out_parts: List[str] = []
                    if v2_parts:
                        out_parts.append(carry_overlap + v2_parts[0])
                        prev = v2_parts[0]
                        for vp in v2_parts[1:]:
                            carry = prev[-self.overlap:] if self.overlap > 0 else ""
                            out_parts.append(carry + vp)
                            prev = vp
                        # After the last V2 sub-chunk, set carry_overlap for next.
                        if self.overlap > 0 and prev:
                            carry_overlap = prev[-self.overlap:]
                    chunks.extend(out_parts)
                    carry_overlap = ""  # already set above, but ensure no double-carry
                i += 1
                continue

            # --- step 3: extract the paragraph group ------------------
            _, last_end = paragraphs[j - 1]
            raw = text[p_start:last_end].strip()
            chunk = carry_overlap + raw if carry_overlap else raw
            chunks.append(chunk)

            # Carry overlap for the next paragraph chunk.
            if self.overlap > 0 and raw:
                # Derive from the *raw* underlying text (before carry_overlap)
                # so the overlap refers to the original paragraph text.
                carry_overlap = raw[-self.overlap:]

            i = j

        if carry_overlap.strip():
            chunks.append(carry_overlap)

        return self._strip_and_filter(chunks)
