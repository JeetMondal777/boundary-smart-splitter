from __future__ import annotations

import re
from typing import List, Optional, Tuple, Union

from .base import BaseSplitter


class StructureSplitter(BaseSplitter):
    """V4: Section/topic-aware text splitter.

    ``chunk_size`` counts **sections** (topics), using document-structure
    signals — headings, numbered sections, transition phrases, and double
    blank lines — to identify topic boundaries.

    When a single section exceeds ``max_chars`` the splitter falls back to
    V3 (paragraph-boundary) behaviour within that section, continuing the
    fallback chain V4 → V3 → V2 → V1 → hard char-cut.

    Parameters
    ----------
    chunk_size : int, default 1
        Target number of sections per chunk.
    max_chars : int, default 1500
        Hard character ceiling.  If a section exceeds this, the section is
        passed to V3 for paragraph-level splitting.
    tolerance : int, default 0
        Section tolerance (reserved for future adjustment).
    respect_headings : bool, default True
        Treat Markdown headings (``#``, ``##``, … ``######`` at line start)
        as hard section boundaries.
    respect_numbered_sections : bool, default True
        Treat numbered list items (``1.``, ``2.``, ``3.`` … at line start)
        as section boundaries.  Handles both ``1. Title`` on one line and
        ``2.`` alone on its own line.
    split_on_transitions : bool, default True
        Treat transition phrases (e.g. ``"However,"``) that appear at the
        start of a paragraph as section boundaries.
    transition_phrases : str | list[str], default ``"default"``
        Either ``"default"`` for the built-in list of English transition
        phrases, or a custom list of phrase strings.
    double_newline_as_boundary : bool, default True
        Treat 3+ consecutive newlines (``\\n\\n\\n``) as a section boundary.
    """

    DEFAULT_TRANSITIONS = [
        "However,",
        "In summary,",
        "That said,",
        "Moving on,",
        "Nevertheless,",
        "In contrast,",
        "Furthermore,",
        "Additionally,",
        "First,",
        "Second,",
        "Third,",
        "Finally,",
        "In conclusion,",
        "On the other hand,",
        "As a result,",
        "For example,",
        "In particular,",
        "More importantly,",
        "Specifically,",
        "Therefore,",
        "Thus,",
        "Meanwhile,",
        "Next,",
        "Now,",
        "So,",
        "But,",
        "Yet,",
        "Still,",
        "Instead,",
    ]

    _HEADING_RE = re.compile(r"^#{1,6} [^\n]+", re.MULTILINE)
    _NUMBERED_SECTION_RE = re.compile(r"^\s*\d+\.(?: [^\n]+|$)", re.MULTILINE)
    _DOUBLE_BLANK_RE = re.compile(r"\n{3,}")

    def __init__(
        self,
        *,
        chunk_size: int = 1,
        max_chars: int = 1500,
        tolerance: int = 0,
        respect_headings: bool = True,
        respect_numbered_sections: bool = True,
        split_on_transitions: bool = True,
        transition_phrases: Union[str, list[str]] = "default",
        double_newline_as_boundary: bool = True,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            max_chars=max_chars,
            tolerance=tolerance,
        )
        self.respect_headings = respect_headings
        self.respect_numbered_sections = respect_numbered_sections
        self.split_on_transitions = split_on_transitions
        self.double_newline_as_boundary = double_newline_as_boundary

        if isinstance(transition_phrases, str):
            if transition_phrases == "default":
                self.transition_phrases = self.DEFAULT_TRANSITIONS
            else:
                raise ValueError(
                    "transition_phrases must be 'default' or a list of strings, "
                    f"got {transition_phrases!r}"
                )
        elif isinstance(transition_phrases, list):
            self.transition_phrases = transition_phrases
        else:
            raise TypeError(
                "transition_phrases must be 'default' or a list of strings, "
                f"got {type(transition_phrases).__name__}"
            )

        self._v3_splitter: Optional["ParagraphSplitter"] = None

    @property
    def _paragraph_splitter(self) -> "ParagraphSplitter":
        if self._v3_splitter is None:
            from .v3_paragraph import ParagraphSplitter

            budget = max(self.chunk_size * 2, 2)
            self._v3_splitter = ParagraphSplitter(
                chunk_size=budget,
                max_chars=self.max_chars,
                tolerance=self.tolerance + 1,
            )
        return self._v3_splitter

    # ── section tokenisation ────────────────────────────────────────

    def _tokenize_sections(self, text: str) -> List[Tuple[int, int]]:
        """Return ``[(start, end), ...]`` for each section in *text*.

        Sections are delimited by structural boundaries: headings,
        transition phrases, and double blank lines.  The start of the
        text is always an implicit boundary.
        """
        if not text:
            return []

        boundaries: set[int] = {0}  # start of text is always a boundary

        # 1. Headings — hard boundaries (high confidence)
        if self.respect_headings:
            for m in self._HEADING_RE.finditer(text):
                boundaries.add(m.start())

        # 2. Numbered sections — e.g. "1. Title", "2." on its own line
        if self.respect_numbered_sections:
            for m in self._NUMBERED_SECTION_RE.finditer(text):
                boundaries.add(m.start())

        # 3. Transition phrases — soft boundaries at paragraph start
        if self.split_on_transitions and self.transition_phrases:
            escaped = [re.escape(p) for p in self.transition_phrases]
            pattern = r"(?:^|\n\n)\s*(" + "|".join(escaped) + r")"
            for m in re.finditer(pattern, text, re.IGNORECASE):
                boundaries.add(m.start(1))

        # 4. Double blank lines — soft boundaries
        if self.double_newline_as_boundary:
            for m in self._DOUBLE_BLANK_RE.finditer(text):
                boundaries.add(m.end())

        sorted_bounds = sorted(boundaries)

        # Build section spans: [boundary[i], boundary[i+1])
        sections: List[Tuple[int, int]] = []
        for i, start in enumerate(sorted_bounds):
            if start >= len(text):
                continue
            end = (
                sorted_bounds[i + 1]
                if i + 1 < len(sorted_bounds)
                else len(text)
            )
            if end > start:
                sections.append((start, end))

        return sections

    # ── splitting logic ─────────────────────────────────────────────

    def split(self, text: str) -> List[str]:
        """Split *text* into section-count-respecting chunks."""
        if not text:
            return []

        sections = self._tokenize_sections(text)
        if not sections:
            return []

        chunks: List[str] = []
        n = len(sections)
        i = 0

        while i < n:
            sec_start = sections[i][0]

            # --- step 1: take up to chunk_size sections -------------
            j = min(i + self.chunk_size, n)
            _, sec_end = sections[j - 1]

            # --- step 2: enforce max_chars (drop sections) ----------
            while j > i:
                _, last_end = sections[j - 1]
                if (last_end - sec_start) <= self.max_chars:
                    break
                j -= 1

            # --- step 2b: single section too long → V3 fallback -----
            if j <= i:
                s, e = sections[i]
                sec_text = text[s:e].strip()
                if sec_text:
                    v3_chunks = self._paragraph_splitter.split(sec_text)
                    chunks.extend(v3_chunks)
                i += 1
                continue

            # --- step 3: extract the section group ------------------
            _, last_end = sections[j - 1]
            chunk = text[sec_start:last_end].strip()
            if chunk:
                chunks.append(chunk)

            i = j

        return self._strip_and_filter(chunks)
