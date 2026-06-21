"""boundary-smart-splitter: A context-aware, boundary-respecting text splitter."""

from .base import Chunk
from .v1_word import WordSplitter
from .v2_sentence import SentenceSplitter
from .v3_paragraph import ParagraphSplitter
from .v4_structure import StructureSplitter
from .heading_detector import (
    HeadingResult,
    HeadingDetector,
    MarkdownHeadingDetector,
    HTMLHeadingDetector,
    PlainTextHeadingDetector,
    select_detector,
)
from .evaluator import Evaluator, EvaluationReport

__version__ = "1.0.1"

__all__ = [
    "Chunk",
    "WordSplitter",
    "SentenceSplitter",
    "ParagraphSplitter",
    "StructureSplitter",
    "HeadingResult",
    "HeadingDetector",
    "MarkdownHeadingDetector",
    "HTMLHeadingDetector",
    "PlainTextHeadingDetector",
    "select_detector",
    "Evaluator",
    "EvaluationReport",
]
