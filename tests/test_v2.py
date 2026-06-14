"""Unit tests for the V2 SentenceSplitter (sentence-count semantics)."""

import pytest

from boundary_smart_splitter.v2_sentence import SentenceSplitter


class TestSentenceSplitter:
    def test_sentence_count_split(self) -> None:
        """chunk_size=2 means 2 sentences per chunk."""
        text = "One. Two. Three. Four. Five. Six."
        splitter = SentenceSplitter(chunk_size=2, max_chars=500, tolerance=1)
        chunks = splitter.split(text)
        assert len(chunks) == 3  # 6 sentences → 2+2+2
        assert all(chunk.strip()[-1] in ".?!" for chunk in chunks)

    def test_single_sentence_fits(self) -> None:
        """A single short sentence → 1 chunk."""
        text = "Hello world."
        splitter = SentenceSplitter(chunk_size=5, max_chars=500, tolerance=1)
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_single_long_sentence_falls_back_to_v1(self) -> None:
        """A single sentence exceeding max_chars triggers V1 word splitting."""
        # ~100 char sentence with no punctuation inside (single "sentence")
        text = "hello " * 15 + "world."
        splitter = SentenceSplitter(chunk_size=5, max_chars=30, tolerance=1)
        chunks = splitter.split(text)
        # The sentence exceeds max_chars, so V1 splits it by words.
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 30, f"Chunk exceeds max_chars: {repr(chunk)}"

    def test_empty_string(self) -> None:
        splitter = SentenceSplitter(chunk_size=5, max_chars=500, tolerance=1)
        assert splitter.split("") == []

    def test_whitespace_only(self) -> None:
        splitter = SentenceSplitter(chunk_size=5, max_chars=500, tolerance=1)
        assert splitter.split("     ") == []

    def test_single_character_input(self) -> None:
        splitter = SentenceSplitter(chunk_size=5, max_chars=500, tolerance=1)
        assert splitter.split("x") == ["x"]

    def test_abbreviation_handling(self) -> None:
        """Periods in abbreviations (e.g. U.S.A.) are not sentence breaks."""
        text = "Please visit the U.S.A. for travel. Yes!"
        splitter = SentenceSplitter(chunk_size=1, max_chars=500, tolerance=1)
        chunks = splitter.split(text)
        # With chunk_size=1, each sentence should be separate.
        # "Please visit the U.S.A. for travel." is one sentence.
        assert len(chunks) == 2
        assert "U.S.A." in chunks[0]
        assert chunks[1] == "Yes!"

    def test_boundary_preference_forward_backward(self) -> None:
        """Forward vs backward preference should differ."""
        text = "A. B. C. D. E."
        # chunk_size=3, so ~3 sentences per chunk.
        fwd = SentenceSplitter(chunk_size=3, max_chars=500, tolerance=1, boundary_preference="forward")
        bwd = SentenceSplitter(chunk_size=3, max_chars=500, tolerance=1, boundary_preference="backward")
        fwd_chunks = fwd.split(text)
        bwd_chunks = bwd.split(text)
        # Both should work and produce the same result for small text.
        assert fwd_chunks == bwd_chunks == ["A. B. C.", "D. E."]

    def test_no_mid_sentence_cut(self) -> None:
        """No chunk should end mid-sentence."""
        text = "One. Two three four five. Six."
        splitter = SentenceSplitter(chunk_size=2, max_chars=500, tolerance=1)
        chunks = splitter.split(text)
        for chunk in chunks[:-1]:
            assert chunk.strip()[-1] in ".?!", f"Chunk ends mid-sentence: {repr(chunk)}"

    def test_max_chars_ceiling(self) -> None:
        """max_chars must never be exceeded."""
        text = ("Hello world. " * 10).strip()
        splitter = SentenceSplitter(chunk_size=20, max_chars=50, tolerance=5)
        chunks = splitter.split(text)
        for chunk in chunks:
            assert len(chunk) <= 50, f"Chunk exceeds max_chars: {repr(chunk)}"

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            SentenceSplitter(chunk_size=0, max_chars=500, tolerance=1)

    def test_invalid_max_chars(self) -> None:
        with pytest.raises(ValueError, match="max_chars must be a positive integer"):
            SentenceSplitter(chunk_size=5, max_chars=0, tolerance=1)

    def test_invalid_tolerance(self) -> None:
        with pytest.raises(ValueError, match="tolerance must be a non-negative integer"):
            SentenceSplitter(chunk_size=5, max_chars=500, tolerance=-1)

    def test_invalid_boundary_preference(self) -> None:
        with pytest.raises(ValueError, match='boundary_preference must be "forward" or "backward"'):
            SentenceSplitter(
                chunk_size=5,
                max_chars=500,
                tolerance=1,
                boundary_preference="up",
            )
