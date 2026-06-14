"""boundary-smart-splitter: A context-aware, boundary-respecting text splitter."""

from .v1_word import WordSplitter
from .v2_sentence import SentenceSplitter
from .v3_paragraph import ParagraphSplitter
from .v4_structure import StructureSplitter

__version__ = "1.0.0"

__all__ = [
    "WordSplitter",
    "SentenceSplitter",
    "ParagraphSplitter",
    "StructureSplitter",
]
