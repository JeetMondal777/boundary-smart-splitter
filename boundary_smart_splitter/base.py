from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseSplitter(ABC):
    """Abstract base class for all splitters.

    Each splitter measures ``chunk_size`` in its **own semantic unit** —
    words, sentences, or paragraphs.  A ``max_chars`` ceiling protects
    embedding-model context limits across all versions.

    Parameters
    ----------
    chunk_size : int
        Target count of semantic units (words / sentences / paragraphs)
        per chunk.
    max_chars : int
        Hard character ceiling — no chunk ever exceeds this length.
    tolerance : int
        Number of semantic units to scan forward or backward when
        adjusting a boundary so it lands on a clean break.
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        max_chars: int,
        tolerance: int,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        if tolerance < 0:
            raise ValueError("tolerance must be a non-negative integer")
        self.chunk_size = chunk_size
        self.max_chars = max_chars
        self.tolerance = tolerance

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """Split *text* into chunks and return them."""
        ...

    def _strip_and_filter(self, chunks: list[str]) -> list[str]:
        """Strip whitespace and remove empty strings from a list of chunks."""
        return [chunk.strip() for chunk in chunks if chunk.strip()]
