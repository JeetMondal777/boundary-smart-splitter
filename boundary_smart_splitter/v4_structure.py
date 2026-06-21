from __future__ import annotations

import re
from typing import Callable, Generator, List, Optional, Tuple, Union

from .base import BaseSplitter, Chunk, Section
from .heading_detector import (
    HeadingDetector,
    MarkdownHeadingDetector,
    HTMLHeadingDetector,
    PlainTextHeadingDetector,
    select_detector,
)


class StructureSplitter(BaseSplitter):
    """V4: Section/topic-aware text splitter.

    ``chunk_size`` counts **sections** (topics), using document-structure
    signals — headings, numbered sections, transition phrases, and double
    blank lines — to identify topic boundaries.

    When a single section exceeds ``max_chars`` (or ``max_tokens``) the splitter falls
    back to V3 (paragraph-boundary) behaviour within that section, continuing the
    fallback chain V4 → V3 → V2 → V1 → hard cut.

    Instances of this class are thread-safe. Each ``split()`` call is stateless.

    Parameters
    ----------
    chunk_size : int, default 1
        Target number of sections per chunk.
    max_chars : int, optional, default 1500
        Hard character ceiling — no chunk ever exceeds this length.
    max_tokens : int, optional
        Hard token ceiling — no chunk ever exceeds this number of tokens.
    min_tokens : int, default 200
        Target minimum number of tokens per chunk. Smaller sibling sections
        are merged up to this budget.
    tolerance : int, default 0
        Section tolerance (reserved for future adjustment).
    respect_headings : bool, default True
        Treat headings detected by the ``heading_detector`` as hard section boundaries.
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
    length_function : str | Callable[[str], int], default "cl100k_base"
        The encoding name, model name, or callable used to calculate token count.
    heading_detector : str | HeadingDetector, default "auto"
        The heading detector strategy. Can be "auto", "markdown", "html", "plain",
        or a HeadingDetector instance.
    orphan_strategy : str, default "merge_backward"
        The strategy for handling orphan sections. Options are "merge_backward",
        "drop", or "tag_only".
    orphan_patterns : list[str] | None, default None
        List of regex patterns to match orphan/boilerplate text.
    overlap_mode : str, default "heading"
        The overlap style: "heading" (injects heading + first sentence context)
        or "tail" (repeats trailing token content).
    overlap_tokens : int, default 30
        The size of overlap in tokens.
    overlap_prefix_template : str, default "[context: {heading} — {first_sentence}]"
        The prefix template for "heading" overlap mode.
    fallback_separators : list[str] | None, default None
        List of separators used sequentially when a section is oversized.
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

    _NUMBERED_SECTION_RE = re.compile(r"^\s*\d+\.(?: [^\n]+|$)", re.MULTILINE)
    _DOUBLE_BLANK_RE = re.compile(r"\n{3,}")

    def __init__(
        self,
        *,
        chunk_size: int = 1,
        max_chars: Optional[int] = 1500,
        max_tokens: Optional[int] = None,
        min_tokens: int = 200,
        tolerance: int = 0,
        respect_headings: bool = True,
        respect_numbered_sections: bool = True,
        split_on_transitions: bool = True,
        transition_phrases: Union[str, list[str]] = "default",
        double_newline_as_boundary: bool = True,
        length_function: Union[str, Callable[[str], int]] = "cl100k_base",
        heading_detector: Union[str, HeadingDetector] = "auto",
        orphan_strategy: str = "merge_backward",
        orphan_patterns: Optional[list[str]] = None,
        overlap_mode: str = "heading",
        overlap_tokens: int = 0,
        overlap_prefix_template: str = "[context: {heading} — {first_sentence}]",
        fallback_separators: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            max_chars=max_chars,
            max_tokens=max_tokens,
            tolerance=tolerance,
            length_function=length_function,
        )
        self.min_tokens = min_tokens
        self.respect_headings = respect_headings
        self.respect_numbered_sections = respect_numbered_sections
        self.split_on_transitions = split_on_transitions
        self.double_newline_as_boundary = double_newline_as_boundary
        self.heading_detector = heading_detector
        
        if orphan_strategy not in ("merge_backward", "drop", "tag_only"):
            raise ValueError('orphan_strategy must be "merge_backward", "drop", or "tag_only"')
        self.orphan_strategy = orphan_strategy

        if orphan_patterns is None:
            self.orphan_patterns = [
                r"^©",
                r"^By\s+[A-Z][a-z]+\s+[A-Z][a-z]+",
                r"^\d+\s*-\s*\d+\s*min\s+read",
                r"^\d+\/\d+\/\d+$",
                r"^(Author|Published|Updated):",
                r"^Table of Contents$",
            ]
        else:
            self.orphan_patterns = orphan_patterns

        if overlap_mode not in ("heading", "tail"):
            raise ValueError('overlap_mode must be "heading" or "tail"')
        self.overlap_mode = overlap_mode
        self.overlap_tokens = overlap_tokens
        self.overlap_prefix_template = overlap_prefix_template

        if fallback_separators is None:
            self.fallback_separators = ["\n\n", "\n", ".", " "]
        else:
            self.fallback_separators = fallback_separators

        if isinstance(transition_phrases, str):
            if transition_phrases == "default":
                self.transition_phrases = tuple(self.DEFAULT_TRANSITIONS)
            else:
                raise ValueError(
                    "transition_phrases must be 'default' or a list of strings, "
                    f"got {transition_phrases!r}"
                )
        elif isinstance(transition_phrases, list):
            self.transition_phrases = tuple(transition_phrases)
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
                max_tokens=self.max_tokens,
                tolerance=self.tolerance + 1,
                length_function=self.length_function,
            )
        return self._v3_splitter

    # ── section tokenisation ────────────────────────────────────────

    def _tokenize_sections(self, text: str) -> List[Section]:
        """Return a list of parsed ``Section`` items in *text*."""
        if not text:
            return []

        boundaries: set[int] = {0}  # start of text is always a boundary

        # 1. Headings — hard boundaries (high confidence)
        headings_by_pos: dict[int, Tuple[int, str]] = {}

        if self.respect_headings:
            if isinstance(self.heading_detector, str):
                if self.heading_detector == "auto":
                    detector = select_detector(text)
                elif self.heading_detector == "markdown":
                    detector = MarkdownHeadingDetector()
                elif self.heading_detector == "html":
                    detector = HTMLHeadingDetector()
                elif self.heading_detector == "plain":
                    detector = PlainTextHeadingDetector()
                else:
                    raise ValueError(f"Unknown heading detector: {self.heading_detector!r}")
            elif isinstance(self.heading_detector, HeadingDetector):
                detector = self.heading_detector
            else:
                raise TypeError(f"Invalid heading detector type: {type(self.heading_detector).__name__}")

            pos = 0
            while pos < len(text):
                next_nl = text.find("\n", pos)
                line_end = next_nl if next_nl != -1 else len(text)
                line = text[pos:line_end]
                
                heading_res = detector.detect(line)
                if heading_res is not None:
                    boundaries.add(pos)
                    headings_by_pos[pos] = (heading_res.level, heading_res.text)
                    
                if next_nl == -1:
                    break
                pos = next_nl + 1

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

        sections: List[Section] = []
        heading_stack: list[str] = []

        for i, start in enumerate(sorted_bounds):
            if start >= len(text):
                continue
            end = (
                sorted_bounds[i + 1]
                if i + 1 < len(sorted_bounds)
                else len(text)
            )
            if end > start:
                sec_text = text[start:end]
                
                # Update heading stack
                heading_text = None
                heading_path = ()
                heading_level = 0
                
                if start in headings_by_pos:
                    level, h_text = headings_by_pos[start]
                    while len(heading_stack) >= level:
                        heading_stack.pop()
                    heading_stack.append(h_text)
                    heading_text = h_text
                    heading_path = tuple(heading_stack)
                    heading_level = level
                else:
                    # Inherit the current path
                    heading_path = tuple(heading_stack)
                    if heading_stack:
                        heading_text = heading_stack[-1]
                        heading_level = len(heading_stack)
                        
                token_count = self._len_fn(sec_text)
                sections.append(
                    Section(
                        text=sec_text,
                        start=start,
                        end=end,
                        heading_text=heading_text,
                        heading_path=heading_path,
                        heading_level=heading_level,
                        token_count=token_count,
                    )
                )

        return sections

    # ── configurable fallback ───────────────────────────────────────

    def _fallback_split(self, text: str, separator_idx: int) -> list[str]:
        """Recursively split text by separators until limits are met."""
        if not text.strip():
            return []

        # Out of separators → hard-cut
        if separator_idx >= len(self.fallback_separators):
            chunks = []
            start = 0
            end = len(text)
            while start < end:
                cut = self._find_cut_point(text, start, end)
                piece = text[start:cut].strip()
                if piece:
                    chunks.append(piece)
                start = cut
            return chunks

        sep = self.fallback_separators[separator_idx]

        if sep == "\n\n":
            from .v3_paragraph import ParagraphSplitter
            splitter = ParagraphSplitter(
                chunk_size=1,
                max_chars=self.max_chars,
                max_tokens=self.max_tokens,
                tolerance=0,
                length_function=self.length_function,
            )
            return splitter.split(text)
        elif sep == ".":
            from .v2_sentence import SentenceSplitter
            splitter = SentenceSplitter(
                chunk_size=1,
                max_chars=self.max_chars,
                max_tokens=self.max_tokens,
                tolerance=0,
                length_function=self.length_function,
            )
            return splitter.split(text)
        elif sep == " ":
            from .v1_word import WordSplitter
            splitter = WordSplitter(
                chunk_size=1,
                max_chars=self.max_chars,
                max_tokens=self.max_tokens,
                tolerance=0,
                length_function=self.length_function,
            )
            return splitter.split(text)
        elif sep == "\n":
            lines = text.split("\n")
            chunks = []
            buffer = []
            buffer_chars = 0
            buffer_tokens = 0
            for line in lines:
                line_chars = len(line)
                line_tokens = self._len_fn(line)
                char_ok = self.max_chars is None or (buffer_chars + line_chars + 1) <= self.max_chars
                token_ok = self.max_tokens is None or (buffer_tokens + line_tokens) <= self.max_tokens
                if char_ok and token_ok:
                    buffer.append(line)
                    buffer_chars += line_chars + 1
                    buffer_tokens += line_tokens
                else:
                    if buffer:
                       chunks.append("\n".join(buffer))
                    if self.max_chars is not None and line_chars > self.max_chars or self.max_tokens is not None and line_tokens > self.max_tokens:
                        chunks.extend(self._fallback_split(line, separator_idx + 1))
                    else:
                        buffer = [line]
                        buffer_chars = line_chars
                        buffer_tokens = line_tokens
            if buffer:
                chunks.append("\n".join(buffer))
            return chunks
        else:
            parts = text.split(sep)
            chunks = []
            buffer = []
            buffer_chars = 0
            buffer_tokens = 0
            for part in parts:
                part_chars = len(part)
                part_tokens = self._len_fn(part)
                char_ok = self.max_chars is None or (buffer_chars + part_chars + len(sep)) <= self.max_chars
                token_ok = self.max_tokens is None or (buffer_tokens + part_tokens) <= self.max_tokens
                if char_ok and token_ok:
                    buffer.append(part)
                    buffer_chars += part_chars + len(sep)
                    buffer_tokens += part_tokens
                else:
                    if buffer:
                        chunks.append(sep.join(buffer))
                    if self.max_chars is not None and part_chars > self.max_chars or self.max_tokens is not None and part_tokens > self.max_tokens:
                        chunks.extend(self._fallback_split(part, separator_idx + 1))
                    else:
                        buffer = [part]
                        buffer_chars = part_chars
                        buffer_tokens = part_tokens
            if buffer:
                chunks.append(sep.join(buffer))
            return chunks

    def _get_first_sentence(self, text: str) -> str:
        # Strip heading lines (e.g. lines starting with '#' or matching standard HTML header tags)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or re.search(r"<h[1-6]\b", stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        cleaned_text = "\n".join(cleaned_lines).strip()

        match = re.search(r"^(.*?[\.\?!])(?:\s|$)", cleaned_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return cleaned_text

    # ── splitting logic ─────────────────────────────────────────────

    def _apply_overlap(self, chunk_text: str, meta: dict, prev_text: str, prev_meta: dict) -> Tuple[str, bool]:
        injected = False
        if self.overlap_mode == "heading":
            prev_heading = " > ".join(prev_meta["heading_path"]) if prev_meta["heading_path"] else "root"
            prev_first_sentence = self._get_first_sentence(prev_text)
            if prev_first_sentence.startswith("[context:"):
                idx_end = prev_first_sentence.find("]")
                if idx_end != -1:
                    prev_first_sentence = prev_first_sentence[idx_end + 1:].strip()
                    prev_first_sentence = self._get_first_sentence(prev_first_sentence)
            
            prefix = self.overlap_prefix_template.format(
                heading=prev_heading,
                first_sentence=prev_first_sentence
            )
            if self.max_tokens is None or self._len_fn(prefix + "\n" + chunk_text) <= self.max_tokens:
                chunk_text = prefix + "\n" + chunk_text
                injected = True
        elif self.overlap_mode == "tail":
            tail_start = len(prev_text)
            while tail_start > 0:
                candidate = prev_text[tail_start - 1:]
                if self._len_fn(candidate) > self.overlap_tokens:
                    break
                tail_start -= 1
            tail = prev_text[tail_start:]
            if tail.strip():
                chunk_text = tail + "\n" + chunk_text
                injected = True
        return chunk_text, injected

    # ── splitting logic ─────────────────────────────────────────────

    def _calculate_quality_score(self, meta: dict) -> float:
        # 1. Boundary Strength (max 0.5)
        boundary_type = meta.get("boundary_type", "soft")
        if boundary_type == "soft":
            boundary_score = 0.5
        elif boundary_type == "merged_sibling":
            boundary_score = 0.4
        elif boundary_type == "hard":
            boundary_score = 0.1
        else:
            boundary_score = 0.3

        # 2. Token Consistency (max 0.5)
        token_count = meta.get("token_count", 0)
        target_min = self.min_tokens
        
        # Determine target max
        if self.max_tokens is not None:
            target_max = self.max_tokens
        elif self.max_chars is not None:
            target_max = self._len_fn("a" * self.max_chars)
        else:
            target_max = 512

        if token_count >= target_min:
            if token_count <= target_max:
                token_score = 0.5
            else:
                token_score = 0.5 * (target_max / token_count) if token_count > 0 else 0.5
                if token_score < 0.1:
                    token_score = 0.1
        else:
            if target_min > 0:
                token_score = 0.5 * (token_count / target_min)
            else:
                token_score = 0.5

        # 3. Deductions
        deductions = 0.0
        if meta.get("orphan_merged"):
            deductions += 0.1
        if meta.get("tail_merged"):
            deductions += 0.05

        score = boundary_score + token_score - deductions
        return max(0.0, min(1.0, score))

    def split_with_metadata_stream(self, text: str) -> Generator[Chunk, None, None]:
        """Yield Chunks of *text* with hierarchical metadata as they are formed."""
        if not text or not text.strip():
            return

        sections = self._tokenize_sections(text)
        if not sections:
            return

        # Step 1: Handle Orphans
        processed_sections: List[Section] = []
        for sec in sections:
            is_orphan = False
            # Check against patterns
            for pat in self.orphan_patterns:
                if re.search(pat, sec.text.strip(), re.MULTILINE | re.IGNORECASE):
                    is_orphan = True
                    break
            
            if is_orphan:
                if self.orphan_strategy == "drop":
                    continue
                elif self.orphan_strategy == "tag_only":
                    continue
                else:  # "merge_backward"
                    if processed_sections:
                        prev = processed_sections[-1]
                        prev.text = (prev.text + "\n\n" + sec.text).strip()
                        prev.end = sec.end
                        prev.token_count = self._len_fn(prev.text)
                        prev.orphan_merged = True
                        continue

            processed_sections.append(sec)

        if not processed_sections:
            return

        # Step 2: Sibling Merging generator
        def generate_merged_chunks():
            buffer: List[Section] = []
            buffer_chars = 0
            buffer_tokens = 0

            def flush(buf: List[Section]):
                if not buf:
                    return None
                merged_text = "\n\n".join(s.text for s in buf).strip()
                if not merged_text:
                    return None
                has_orphan = any(s.orphan_merged for s in buf)
                meta = {
                    "section_title": buf[0].heading_text,
                    "heading_path": list(buf[0].heading_path),
                    "heading_level": buf[0].heading_level,
                    "token_count": buffer_tokens,
                    "char_count": len(merged_text),
                    "boundary_type": "merged_sibling" if len(buf) > 1 else "soft",
                    "overlap_injected": False,
                    "source_format": "html" if "html" in str(self.heading_detector).lower() else "markdown",
                    "orphan_merged": has_orphan,
                }
                return (merged_text, meta)

            for sec in processed_sections:
                if not buffer:
                    buffer = [sec]
                    buffer_chars = len(sec.text)
                    buffer_tokens = sec.token_count
                    continue

                char_ok = self.max_chars is None or (buffer_chars + len(sec.text) + 2) <= self.max_chars
                token_ok = self.max_tokens is None or (buffer_tokens + sec.token_count) <= self.max_tokens

                is_sibling = sec.heading_path[:-1] == buffer[0].heading_path[:-1]

                if sec.heading_level == 1:
                    can_group = (
                        len(buffer) < self.chunk_size 
                        and buffer[0].heading_level == 1
                    )
                else:
                    can_group = (len(buffer) < self.chunk_size) or (is_sibling and buffer_tokens < self.min_tokens)

                if can_group and char_ok and token_ok:
                    buffer.append(sec)
                    buffer_chars += len(sec.text) + 2
                    buffer_tokens += sec.token_count
                else:
                    item = flush(buffer)
                    if item:
                        yield item
                    buffer = [sec]
                    buffer_chars = len(sec.text)
                    buffer_tokens = sec.token_count

            item = flush(buffer)
            if item:
                yield item

        # Step 3: Fallback generator
        def generate_fallback_chunks():
            for text_val, meta in generate_merged_chunks():
                char_exceeded = self.max_chars is not None and len(text_val) > self.max_chars
                token_exceeded = self.max_tokens is not None and meta["token_count"] > self.max_tokens
                if char_exceeded or token_exceeded:
                    sub_pieces = self._fallback_split(text_val, 0)
                    for sp in sub_pieces:
                        sp_meta = meta.copy()
                        sp_meta["token_count"] = self._len_fn(sp)
                        sp_meta["char_count"] = len(sp)
                        sp_meta["boundary_type"] = "hard"
                        sp_meta["forced_fallback"] = True
                        yield (sp, sp_meta)
                else:
                    yield (text_val, meta)

        # Step 4: Trailing tail check generator
        def generate_tail_checked_chunks():
            prev_item = None
            for curr_text, curr_meta in generate_fallback_chunks():
                if prev_item is None:
                    prev_item = (curr_text, curr_meta)
                    continue

                prev_text, prev_meta = prev_item
                if curr_meta["token_count"] < self.min_tokens:
                    if prev_meta.get("heading_level") == 1 or curr_meta.get("heading_level") == 1:
                        yield prev_item
                        prev_item = (curr_text, curr_meta)
                        continue

                    same_top = False
                    if curr_meta["heading_path"] and prev_meta["heading_path"]:
                        same_top = curr_meta["heading_path"][0] == prev_meta["heading_path"][0]
                    elif not curr_meta["heading_path"] and not prev_meta["heading_path"]:
                        same_top = True

                    new_chars = len(prev_text) + 2 + len(curr_text)
                    new_tokens = prev_meta["token_count"] + curr_meta["token_count"]
                    char_ok = self.max_chars is None or new_chars <= self.max_chars
                    token_ok = self.max_tokens is None or new_tokens <= self.max_tokens

                    # Avoid merging a pure heading backward into a previous chunk
                    def is_pure_heading(t: str) -> bool:
                        lines = [line.strip() for line in t.split("\n") if line.strip()]
                        if not lines:
                            return False
                        for line in lines:
                            if not (line.startswith("#") or re.match(r"^<h[1-6]\b", line, re.IGNORECASE)):
                                return False
                        return True

                    if same_top and char_ok and token_ok and not is_pure_heading(curr_text):
                        merged_text = prev_text + "\n\n" + curr_text
                        prev_meta["token_count"] = new_tokens
                        prev_meta["char_count"] = len(merged_text)
                        prev_meta["boundary_type"] = "merged_sibling"
                        prev_meta["tail_merged"] = True
                        
                        if curr_meta.get("forced_fallback"):
                            prev_meta["forced_fallback"] = True
                        if curr_meta.get("orphan_merged"):
                            prev_meta["orphan_merged"] = True
                        if curr_meta.get("tail_merged"):
                            prev_meta["tail_merged"] = True

                        prev_item = (merged_text, prev_meta)
                        continue

                yield prev_item
                prev_item = (curr_text, curr_meta)

            if prev_item is not None:
                yield prev_item

        # Step 5: Final output and overlap generator (with 1-item lookahead for doc position)
        prev_item = None
        last_raw_text = None
        last_meta = None
        chunk_idx = 0

        for curr_text, curr_meta in generate_tail_checked_chunks():
            if prev_item is not None:
                p_text, p_meta = prev_item
                injected = False
                if self.overlap_tokens > 0 and chunk_idx > 0 and last_raw_text is not None and last_meta is not None:
                    p_text, injected = self._apply_overlap(p_text, p_meta, last_raw_text, last_meta)

                p_meta["overlap_injected"] = injected
                p_meta["section_index"] = chunk_idx
                p_meta["position_in_doc"] = "start" if chunk_idx == 0 else "middle"
                p_meta["token_count"] = self._len_fn(p_text)
                p_meta["char_count"] = len(p_text)

                # Now calculate quality metrics
                quality_flags = []
                if p_meta["token_count"] < self.min_tokens:
                    quality_flags.append("under_min_tokens")
                if p_meta.get("forced_fallback"):
                    quality_flags.append("forced_fallback")
                if p_meta.get("orphan_merged"):
                    quality_flags.append("orphan_merged")
                if p_meta.get("tail_merged"):
                    quality_flags.append("tail_merged")

                p_meta["quality_flags"] = quality_flags
                p_meta["quality_score"] = self._calculate_quality_score(p_meta)

                yield Chunk(text=p_text, metadata=p_meta)

                last_raw_text = prev_item[0]
                last_meta = prev_item[1]
                chunk_idx += 1

            prev_item = (curr_text, curr_meta)

        if prev_item is not None:
            p_text, p_meta = prev_item
            injected = False
            if self.overlap_tokens > 0 and chunk_idx > 0 and last_raw_text is not None and last_meta is not None:
                p_text, injected = self._apply_overlap(p_text, p_meta, last_raw_text, last_meta)

            p_meta["overlap_injected"] = injected
            p_meta["section_index"] = chunk_idx
            p_meta["position_in_doc"] = "start" if chunk_idx == 0 else "end"
            p_meta["token_count"] = self._len_fn(p_text)
            p_meta["char_count"] = len(p_text)

            # Now calculate quality metrics
            quality_flags = []
            if p_meta["token_count"] < self.min_tokens:
                quality_flags.append("under_min_tokens")
            if p_meta.get("forced_fallback"):
                quality_flags.append("forced_fallback")
            if p_meta.get("orphan_merged"):
                quality_flags.append("orphan_merged")
            if p_meta.get("tail_merged"):
                quality_flags.append("tail_merged")

            p_meta["quality_flags"] = quality_flags
            p_meta["quality_score"] = self._calculate_quality_score(p_meta)

            yield Chunk(text=p_text, metadata=p_meta)

    def split_with_metadata(self, text: str) -> List[Chunk]:
        """Split text and return chunks with hierarchical metadata."""
        return list(self.split_with_metadata_stream(text))

    def split_stream(self, text: str) -> Generator[str, None, None]:
        """Yield chunks of *text* as they are formed."""
        for chunk in self.split_with_metadata_stream(text):
            yield chunk.text

    def split(self, text: str) -> List[str]:
        """Split *text* into section-count-respecting chunks."""
        return list(self.split_stream(text))
