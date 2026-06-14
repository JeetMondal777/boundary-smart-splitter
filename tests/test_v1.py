"""Unit tests for the V1 WordSplitter (word-count semantics)."""

import pytest

from boundary_smart_splitter.v1_word import WordSplitter


class TestWordSplitter:
    def test_word_count_split(self) -> None:
        """chunk_size=3 means exactly 3 words per chunk."""
        text = "one two three four five six seven eight"
        splitter = WordSplitter(chunk_size=3, max_chars=500, tolerance=1)
        chunks = splitter.split(text)
        # Each chunk (except last) should have 3 words.
        assert len(chunks) == 3  # 8 words → 3+3+2
        assert chunks[0] == "one two three"
        assert chunks[1] == "four five six"
        assert chunks[2] == "seven eight"

    def test_max_chars_ceiling(self) -> None:
        """max_chars must never be exceeded even if chunk_size says more."""
        text = ("hello " * 100) + "world"
        splitter = WordSplitter(chunk_size=100, max_chars=30, tolerance=1)
        chunks = splitter.split(text)
        for chunk in chunks:
            assert len(chunk) <= 30, f"Chunk exceeds max_chars: {repr(chunk)}"

    def test_empty_string(self) -> None:
        splitter = WordSplitter(chunk_size=10, max_chars=100, tolerance=1)
        assert splitter.split("") == []

    def test_whitespace_only(self) -> None:
        splitter = WordSplitter(chunk_size=10, max_chars=100, tolerance=1)
        assert splitter.split("     ") == []

    def test_single_character_input(self) -> None:
        splitter = WordSplitter(chunk_size=10, max_chars=100, tolerance=1)
        assert splitter.split("x") == ["x"]

    def test_single_long_word(self) -> None:
        """Single word exceeding max_chars must be hard-cut at max_chars boundaries."""
        text = "x" * 500
        splitter = WordSplitter(chunk_size=1, max_chars=30, tolerance=1)
        chunks = splitter.split(text)
        # max_chars=30 → a 500-char word gets split into 30-char pieces + tail
        assert len(chunks) == 17  # 500 / 30 ≈ 16.67 → 17 chunks
        for chunk in chunks[:-1]:
            assert len(chunk) <= 30
        assert sum(len(c) for c in chunks) == 500

    def test_tolerance_adjusts_boundary(self) -> None:
        """Tolerance allows scanning extra chars to land on a clean boundary."""
        text = "a b c d e f gggggggg h i j"
        splitter = WordSplitter(chunk_size=4, max_chars=500, tolerance=1)
        chunks = splitter.split(text)
        # 4 words per chunk, no mid-word cut because we split at word boundaries.
        assert chunks[0] == "a b c d"
        assert chunks[1] == "e f gggggggg h"
        assert chunks[2] == "i j"

    def test_word_count_never_exceeds(self) -> None:
        """Word count per chunk should be <= chunk_size (except last)."""
        text = ("word " * 50).strip()
        splitter = WordSplitter(chunk_size=10, max_chars=500, tolerance=1)
        chunks = splitter.split(text)
        for i, chunk in enumerate(chunks[:-1]):
            word_count = len(chunk.split())
            assert word_count <= 10, (
                f"Chunk {i} has {word_count} words, expected ≤10"
            )

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            WordSplitter(chunk_size=0, max_chars=100, tolerance=1)

    def test_invalid_max_chars(self) -> None:
        with pytest.raises(ValueError, match="max_chars must be a positive integer"):
            WordSplitter(chunk_size=10, max_chars=0, tolerance=1)

    def test_invalid_tolerance(self) -> None:
        with pytest.raises(ValueError, match="tolerance must be a non-negative integer"):
            WordSplitter(chunk_size=10, max_chars=100, tolerance=-1)
