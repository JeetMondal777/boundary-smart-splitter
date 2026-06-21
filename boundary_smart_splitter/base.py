from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generator, List, Optional, Union
import warnings

class BoundarySmartSplitterError(ValueError):
    """Custom error class for boundary-smart-splitter validation errors."""
    pass


@dataclass
class Chunk:
    text: str
    metadata: dict


@dataclass
class Section:
    text: str
    start: int
    end: int
    heading_text: Optional[str] = None
    heading_path: tuple[str, ...] = ()
    heading_level: int = 0
    token_count: int = 0
    orphan_merged: bool = False


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
    max_chars : int, optional
        Hard character ceiling — no chunk ever exceeds this length.
    max_tokens : int, optional
        Hard token ceiling — no chunk ever exceeds this number of tokens.
    tolerance : int
        Number of semantic units to scan forward or backward when
        adjusting a boundary so it lands on a clean break.
    length_function : str | Callable[[str], int], default "cl100k_base"
        The encoding name, model name, or callable used to calculate token count.
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        max_chars: Optional[int] = 1500,
        max_tokens: Optional[int] = None,
        tolerance: int,
        length_function: Union[str, Callable[[str], int]] = "cl100k_base",
    ) -> None:
        if chunk_size <= 0:
            raise BoundarySmartSplitterError(
                f"Invalid chunk_size: {chunk_size}. "
                "chunk_size must be a positive integer representing the target number of units (e.g. sections/paragraphs/sentences/words). "
                "For example, a sensible default is 1 (for StructureSplitter) or 5 (for SentenceSplitter)."
            )
        if max_chars is not None and max_chars <= 0:
            raise BoundarySmartSplitterError(
                f"Invalid max_chars: {max_chars}. "
                "max_chars must be a positive integer acting as a hard character ceiling. "
                "For example, a sensible default is 1500 characters."
            )
        if max_tokens is not None and max_tokens <= 0:
            raise BoundarySmartSplitterError(
                f"Invalid max_tokens: {max_tokens}. "
                "max_tokens must be a positive integer acting as a hard token ceiling. "
                "For example, a sensible default is 512 tokens."
            )
        if tolerance < 0:
            raise BoundarySmartSplitterError(
                f"Invalid tolerance: {tolerance}. "
                "tolerance must be a non-negative integer representing the search range. "
                "For example, a sensible default is 2 (for SentenceSplitter) or 10 (for WordSplitter)."
            )
        self.chunk_size = chunk_size
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.tolerance = tolerance
        self.length_function = length_function

        # Initialize the length function
        if isinstance(length_function, str):
            try:
                import tiktoken
                try:
                    enc = tiktoken.get_encoding(length_function)
                    self._len_fn = lambda text: len(enc.encode(text))
                except (ValueError, KeyError):
                    try:
                        enc = tiktoken.encoding_for_model(length_function)
                        self._len_fn = lambda text: len(enc.encode(text))
                    except (ValueError, KeyError) as e:
                        raise BoundarySmartSplitterError(
                            f"Unsupported encoding or model name: {length_function!r}"
                        ) from e
            except ImportError:
                warnings.warn(
                    "tiktoken is not installed, falling back to character-length counting. "
                    "To enable token-aware splitting, run 'pip install tiktoken'.",
                    ImportWarning,
                    stacklevel=2,
                )
                self._len_fn = len
        elif callable(length_function):
            self._len_fn = length_function
        else:
            raise TypeError(
                f"length_function must be a string or a callable, got {type(length_function).__name__}"
            )

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """Split *text* into chunks and return them."""
        ...

    @abstractmethod
    def split_stream(self, text: str) -> Generator[str, None, None]:
        """Yield chunks of *text* as they are formed."""
        ...

    def _strip_and_filter(self, chunks: list[str]) -> list[str]:
        """Strip whitespace and remove empty strings from a list of chunks."""
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _find_cut_point(self, text: str, start: int, end: int) -> int:
        """Find the maximum end index <= `end` starting from `start` such that
        the substring satisfies both max_chars and max_tokens bounds.
        """
        # Respect max_chars first (easy slice)
        limit_end = end
        if self.max_chars is not None:
            limit_end = min(limit_end, start + self.max_chars)

        if self.max_tokens is None:
            return limit_end

        # Binary search for max_tokens
        low = start + 1
        high = limit_end
        ans = start + 1  # must take at least 1 character to avoid infinite loop

        while low <= high:
            mid = (low + high) // 2
            substring = text[start:mid]
            if self._len_fn(substring) <= self.max_tokens:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans
