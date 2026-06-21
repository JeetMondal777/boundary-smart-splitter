"""LangChain compatibility for boundary-smart-splitter."""

try:
    from langchain_text_splitters import TextSplitter as _LangChainTextSplitter
except ImportError as exc:
    raise ImportError(
        "LangChain integration requires the 'langchain-text-splitters' package. "
        "Install it with: pip install langchain-text-splitters"
    ) from exc

from typing import Callable, List, Optional, Union

from ..v1_word import WordSplitter
from ..v2_sentence import SentenceSplitter
from ..v3_paragraph import ParagraphSplitter
from ..v4_structure import StructureSplitter
from ..heading_detector import HeadingDetector


class LangChainWordSplitter(_LangChainTextSplitter):
    """LangChain-compatible wrapper around :class:`WordSplitter`.

    Parameters
    ----------
    chunk_size : int, default 60
        Target number of words per chunk. If > 10, dynamically mapped to max_chars ceiling.
    chunk_overlap : int, default 0
        Standard parameter (ignored by word-level splitter, but preserved for LangChain).
    max_chars : int, default 500
        Hard character ceiling.
    max_tokens : int, optional
        Hard token ceiling.
    tolerance : int, default 10
        Word tolerance for boundary adjustment.
    length_function : str | Callable[[str], int], default "cl100k_base"
        The encoding name, model name, or callable used to calculate token count.
    **kwargs :
        Passed to the base ``TextSplitter`` constructor.
    """

    def __init__(
        self,
        chunk_size: int = 60,
        chunk_overlap: int = 0,
        max_chars: Optional[int] = 500,
        max_tokens: Optional[int] = None,
        tolerance: int = 10,
        length_function: Union[str, Callable[[str], int]] = "cl100k_base",
        **kwargs,
    ):
        lc_len_fn = len
        if callable(length_function):
            lc_len_fn = length_function

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lc_len_fn,
            **kwargs,
        )

        default_chunk_size = 60
        actual_chunk_size = chunk_size
        resolved_max_chars = max_chars
        resolved_max_tokens = max_tokens

        if chunk_size > 10:
            actual_chunk_size = default_chunk_size
            if max_tokens is not None:
                resolved_max_tokens = chunk_size
            else:
                resolved_max_chars = chunk_size

        self._splitter = WordSplitter(
            chunk_size=actual_chunk_size,
            max_chars=resolved_max_chars,
            max_tokens=resolved_max_tokens,
            tolerance=tolerance,
            length_function=length_function,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split(text)


class LangChainSentenceSplitter(_LangChainTextSplitter):
    """LangChain-compatible wrapper around :class:`SentenceSplitter`.

    Parameters
    ----------
    chunk_size : int, default 5
        Target number of sentences per chunk. If > 10, dynamically mapped to max_chars ceiling.
    chunk_overlap : int, default 0
        Standard parameter (ignored by sentence-level splitter, but preserved for LangChain).
    max_chars : int, default 500
        Hard character ceiling.
    max_tokens : int, optional
        Hard token ceiling.
    tolerance : int, default 2
        Sentence tolerance for boundary adjustment.
    boundary_preference : str, default "forward"
        ``"forward"`` or ``"backward"``.
    abbreviations : set[str] | None, optional
        Custom abbreviation set.
    length_function : str | Callable[[str], int], default "cl100k_base"
        The encoding name, model name, or callable used to calculate token count.
    **kwargs :
        Passed to the base ``TextSplitter`` constructor.
    """

    def __init__(
        self,
        chunk_size: int = 5,
        chunk_overlap: int = 0,
        max_chars: Optional[int] = 500,
        max_tokens: Optional[int] = None,
        tolerance: int = 2,
        boundary_preference: str = "forward",
        abbreviations: Optional[set[str]] = None,
        length_function: Union[str, Callable[[str], int]] = "cl100k_base",
        **kwargs,
    ):
        lc_len_fn = len
        if callable(length_function):
            lc_len_fn = length_function

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lc_len_fn,
            **kwargs,
        )

        default_chunk_size = 5
        actual_chunk_size = chunk_size
        resolved_max_chars = max_chars
        resolved_max_tokens = max_tokens

        if chunk_size > 10:
            actual_chunk_size = default_chunk_size
            if max_tokens is not None:
                resolved_max_tokens = chunk_size
            else:
                resolved_max_chars = chunk_size

        self._splitter = SentenceSplitter(
            chunk_size=actual_chunk_size,
            max_chars=resolved_max_chars,
            max_tokens=resolved_max_tokens,
            tolerance=tolerance,
            boundary_preference=boundary_preference,
            abbreviations=abbreviations,
            length_function=length_function,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split(text)


class LangChainParagraphSplitter(_LangChainTextSplitter):
    """LangChain-compatible wrapper around :class:`ParagraphSplitter`.

    Parameters
    ----------
    chunk_size : int, default 2
        Target number of paragraphs per chunk. If > 10, dynamically mapped to max_chars ceiling.
    chunk_overlap : int, default 0
        Mapped to ``overlap`` (characters) unless ``overlap`` is explicitly provided.
    max_chars : int, default 500
        Hard character ceiling.
    max_tokens : int, optional
        Hard token ceiling.
    tolerance : int, default 1
        Paragraph tolerance for boundary adjustment.
    overlap : int, optional
        Characters of context overlap between consecutive chunks.
    paragraph_separator : str, default ``"\\n\\n"``
        String that separates paragraphs.
    use_markdown_mode : bool, default False
        Treat horizontal rules and headings as paragraph boundaries.
    boundary_preference : str, default "forward"
        Passed to internal SentenceSplitter for long-paragraph fallback.
    abbreviations : set[str] | None, optional
        Passed to internal SentenceSplitter.
    length_function : str | Callable[[str], int], default "cl100k_base"
        The encoding name, model name, or callable used to calculate token count.
    **kwargs :
        Passed to the base ``TextSplitter`` constructor.
    """

    def __init__(
        self,
        chunk_size: int = 2,
        chunk_overlap: int = 0,
        max_chars: Optional[int] = 500,
        max_tokens: Optional[int] = None,
        tolerance: int = 1,
        overlap: Optional[int] = None,
        paragraph_separator: str = "\n\n",
        use_markdown_mode: bool = False,
        boundary_preference: str = "forward",
        abbreviations: Optional[set[str]] = None,
        length_function: Union[str, Callable[[str], int]] = "cl100k_base",
        **kwargs,
    ):
        lc_len_fn = len
        if callable(length_function):
            lc_len_fn = length_function

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lc_len_fn,
            **kwargs,
        )

        default_chunk_size = 2
        actual_chunk_size = chunk_size
        resolved_max_chars = max_chars
        resolved_max_tokens = max_tokens

        if chunk_size > 10:
            actual_chunk_size = default_chunk_size
            if max_tokens is not None:
                resolved_max_tokens = chunk_size
            else:
                resolved_max_chars = chunk_size

        resolved_overlap = overlap if overlap is not None else chunk_overlap

        self._splitter = ParagraphSplitter(
            chunk_size=actual_chunk_size,
            max_chars=resolved_max_chars,
            max_tokens=resolved_max_tokens,
            tolerance=tolerance,
            overlap=resolved_overlap,
            paragraph_separator=paragraph_separator,
            use_markdown_mode=use_markdown_mode,
            boundary_preference=boundary_preference,
            abbreviations=abbreviations,
            length_function=length_function,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split(text)


class LangChainStructureSplitter(_LangChainTextSplitter):
    """LangChain-compatible wrapper around :class:`StructureSplitter`.

    Parameters
    ----------
    chunk_size : int, default 1
        Target number of sections per chunk. If > 10, dynamically mapped to max_chars ceiling.
    chunk_overlap : int, default 0
        Mapped to ``overlap_tokens`` unless ``overlap_tokens`` is explicitly provided.
    max_chars : int, default 1500
        Hard character ceiling — falls back to V3 if a section exceeds this.
    max_tokens : int, optional
        Hard token ceiling.
    min_tokens : int, default 200
        Target minimum number of tokens per chunk.
    tolerance : int, default 0
        Section tolerance (reserved for future use).
    respect_headings : bool, default True
        Treat Markdown headings (``#``, ``##``, …) as hard boundaries.
    respect_numbered_sections : bool, default True
        Treat numbered list items as section boundaries.
    split_on_transitions : bool, default True
        Treat transition phrases at paragraph start as boundaries.
    transition_phrases : str | list[str], default ``"default"``
        Built-in or custom transition phrase list.
    double_newline_as_boundary : bool, default True
        Treat 3+ consecutive newlines as a section boundary.
    heading_detector : str | HeadingDetector, default "auto"
        The heading detector strategy.
    orphan_strategy : str, default "merge_backward"
        The strategy for handling orphan sections.
    orphan_patterns : list[str] | None, default None
        List of regex patterns to match orphan/boilerplate text.
    overlap_mode : str, default "heading"
        The overlap style: "heading" or "tail".
    overlap_tokens : int, optional
        The size of overlap in tokens.
    overlap_prefix_template : str, default "[context: {heading} — {first_sentence}]"
        The prefix template for "heading" overlap mode.
    fallback_separators : list[str] | None, default None
        List of separators used sequentially when a section is oversized.
    length_function : str | Callable[[str], int], default "cl100k_base"
        The encoding name, model name, or callable used to calculate token count.
    **kwargs :
        Passed to the base ``TextSplitter`` constructor.
    """

    def __init__(
        self,
        chunk_size: int = 1,
        chunk_overlap: int = 0,
        max_chars: Optional[int] = 1500,
        max_tokens: Optional[int] = None,
        min_tokens: int = 200,
        tolerance: int = 0,
        respect_headings: bool = True,
        respect_numbered_sections: bool = True,
        split_on_transitions: bool = True,
        transition_phrases: Union[str, list[str]] = "default",
        double_newline_as_boundary: bool = True,
        heading_detector: Union[str, HeadingDetector] = "auto",
        orphan_strategy: str = "merge_backward",
        orphan_patterns: Optional[list[str]] = None,
        overlap_mode: str = "heading",
        overlap_tokens: Optional[int] = None,
        overlap_prefix_template: str = "[context: {heading} — {first_sentence}]",
        fallback_separators: Optional[list[str]] = None,
        length_function: Union[str, Callable[[str], int]] = "cl100k_base",
        **kwargs,
    ):
        lc_len_fn = len
        if callable(length_function):
            lc_len_fn = length_function

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lc_len_fn,
            **kwargs,
        )

        default_chunk_size = 1
        actual_chunk_size = chunk_size
        resolved_max_chars = max_chars
        resolved_max_tokens = max_tokens

        if chunk_size > 10:
            actual_chunk_size = default_chunk_size
            if max_tokens is not None:
                resolved_max_tokens = chunk_size
            else:
                resolved_max_chars = chunk_size

        resolved_overlap_tokens = overlap_tokens if overlap_tokens is not None else chunk_overlap

        self._splitter = StructureSplitter(
            chunk_size=actual_chunk_size,
            max_chars=resolved_max_chars,
            max_tokens=resolved_max_tokens,
            min_tokens=min_tokens,
            tolerance=tolerance,
            respect_headings=respect_headings,
            respect_numbered_sections=respect_numbered_sections,
            split_on_transitions=split_on_transitions,
            transition_phrases=transition_phrases,
            double_newline_as_boundary=double_newline_as_boundary,
            heading_detector=heading_detector,
            orphan_strategy=orphan_strategy,
            orphan_patterns=orphan_patterns,
            overlap_mode=overlap_mode,
            overlap_tokens=resolved_overlap_tokens,
            overlap_prefix_template=overlap_prefix_template,
            fallback_separators=fallback_separators,
            length_function=length_function,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split(text)
