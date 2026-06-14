"""LangChain compatibility module for boundary-smart-splitter."""

from .compat import (
    LangChainParagraphSplitter,
    LangChainSentenceSplitter,
    LangChainStructureSplitter,
    LangChainWordSplitter,
)

__all__ = [
    "LangChainWordSplitter",
    "LangChainSentenceSplitter",
    "LangChainParagraphSplitter",
    "LangChainStructureSplitter",
]
