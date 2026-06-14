"""LangChain compatibility for boundary-smart-splitter."""

try:
    from langchain_text_splitters import TextSplitter as _LangChainTextSplitter
except ImportError as exc:
    raise ImportError(
        "LangChain integration requires the 'langchain-text-splitters' package. "
        "Install it with: pip install langchain-text-splitters"
    ) from exc

from typing import List, Optional

from ..v1_word import WordSplitter
from ..v2_sentence import SentenceSplitter
from ..v3_paragraph import ParagraphSplitter
from ..v4_structure import StructureSplitter


class LangChainWordSplitter(_LangChainTextSplitter):
    """LangChain-compatible wrapper around :class:`WordSplitter`.

    Parameters
    ----------
    chunk_size : int, default 60
        Target number of words per chunk.
    max_chars : int, default 500
        Hard character ceiling.
    tolerance : int, default 10
        Word tolerance for boundary adjustment.
    **kwargs :
        Passed to the base ``TextSplitter`` constructor.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 60,
        max_chars: int = 500,
        tolerance: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._splitter = WordSplitter(
            chunk_size=chunk_size,
            max_chars=max_chars,
            tolerance=tolerance,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split(text)


class LangChainSentenceSplitter(_LangChainTextSplitter):
    """LangChain-compatible wrapper around :class:`SentenceSplitter`.

    Parameters
    ----------
    chunk_size : int, default 5
        Target number of sentences per chunk.
    max_chars : int, default 500
        Hard character ceiling.
    tolerance : int, default 2
        Sentence tolerance for boundary adjustment.
    boundary_preference : str, default "forward"
        ``"forward"`` or ``"backward"``.
    abbreviations : set[str] | None, optional
        Custom abbreviation set.
    **kwargs :
        Passed to the base ``TextSplitter`` constructor.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 5,
        max_chars: int = 500,
        tolerance: int = 2,
        boundary_preference: str = "forward",
        abbreviations: set[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            max_chars=max_chars,
            tolerance=tolerance,
            boundary_preference=boundary_preference,
            abbreviations=abbreviations,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split(text)


class LangChainParagraphSplitter(_LangChainTextSplitter):
    """LangChain-compatible wrapper around :class:`ParagraphSplitter`.

    Parameters
    ----------
    chunk_size : int, default 2
        Target number of paragraphs per chunk.
    max_chars : int, default 500
        Hard character ceiling.
    tolerance : int, default 1
        Paragraph tolerance for boundary adjustment.
    overlap : int, default 0
        Characters of context overlap between consecutive chunks.
    paragraph_separator : str, default ``"\\n\\n"``
        String that separates paragraphs.
    use_markdown_mode : bool, default False
        Treat horizontal rules and headings as paragraph boundaries.
    boundary_preference : str, default "forward"
        Passed to internal SentenceSplitter for long-paragraph fallback.
    abbreviations : set[str] | None, optional
        Passed to internal SentenceSplitter.
    **kwargs :
        Passed to the base ``TextSplitter`` constructor.
    """

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._splitter = ParagraphSplitter(
            chunk_size=chunk_size,
            max_chars=max_chars,
            tolerance=tolerance,
            overlap=overlap,
            paragraph_separator=paragraph_separator,
            use_markdown_mode=use_markdown_mode,
            boundary_preference=boundary_preference,
            abbreviations=abbreviations,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split(text)


class LangChainStructureSplitter(_LangChainTextSplitter):
    """LangChain-compatible wrapper around :class:`StructureSplitter`.

    ``chunk_size`` counts **sections** (topics).  Uses document-structure
    signals — headings, transition phrases, and double blank lines — to
    identify topic boundaries.

    Parameters
    ----------
    chunk_size : int, default 1
        Target number of sections per chunk.
    max_chars : int, default 1500
        Hard character ceiling — falls back to V3 if a section exceeds this.
    tolerance : int, default 0
        Section tolerance (reserved for future use).
    respect_headings : bool, default True
        Treat Markdown headings (``#``, ``##``, …) as hard boundaries.
    respect_numbered_sections : bool, default True
        Treat numbered list items (``1.``, ``2.``, ``3.`` … at line start)
        as section boundaries.
    split_on_transitions : bool, default True
        Treat transition phrases at paragraph start as boundaries.
    transition_phrases : str | list[str], default ``"default"``
        Built-in or custom transition phrase list.
    double_newline_as_boundary : bool, default True
        Treat 3+ consecutive newlines as a section boundary.
    **kwargs :
        Passed to the base ``TextSplitter`` constructor.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1,
        max_chars: int = 1500,
        tolerance: int = 0,
        respect_headings: bool = True,
        respect_numbered_sections: bool = True,
        split_on_transitions: bool = True,
        transition_phrases: str | list[str] = "default",
        double_newline_as_boundary: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._splitter = StructureSplitter(
            chunk_size=chunk_size,
            max_chars=max_chars,
            tolerance=tolerance,
            respect_headings=respect_headings,
            respect_numbered_sections=respect_numbered_sections,
            split_on_transitions=split_on_transitions,
            transition_phrases=transition_phrases,
            double_newline_as_boundary=double_newline_as_boundary,
        )

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split(text)
