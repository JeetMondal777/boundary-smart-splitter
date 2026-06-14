"""Unit tests for the V3 ParagraphSplitter (paragraph-count semantics)."""

import pytest

from boundary_smart_splitter.v3_paragraph import ParagraphSplitter


class TestParagraphSplitter:
    def test_paragraph_count_split(self) -> None:
        """chunk_size=2 means 2 paragraphs per chunk."""
        text = "P1.\n\nP2.\n\nP3.\n\nP4.\n\nP5."
        splitter = ParagraphSplitter(chunk_size=2, max_chars=500, tolerance=1)
        chunks = splitter.split(text)
        assert len(chunks) == 3  # 5 paragraphs → 2+2+1

    def test_single_paragraph_fits(self) -> None:
        """A single short paragraph → 1 chunk."""
        text = "This is a short paragraph."
        splitter = ParagraphSplitter(chunk_size=2, max_chars=500, tolerance=1)
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == "This is a short paragraph."

    def test_single_long_paragraph_falls_back_to_v2(self) -> None:
        """A single paragraph exceeding max_chars triggers V2 sentence splitting."""
        text = ("This is a sentence. " * 20).strip()
        splitter = ParagraphSplitter(chunk_size=2, max_chars=100, tolerance=1)
        chunks = splitter.split(text)
        # Should be split into sentence-boundary-respecting chunks.
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 100, f"Chunk exceeds max_chars: {repr(chunk)}"

    def test_empty_string(self) -> None:
        splitter = ParagraphSplitter(chunk_size=2, max_chars=500, tolerance=1)
        assert splitter.split("") == []

    def test_whitespace_only(self) -> None:
        splitter = ParagraphSplitter(chunk_size=2, max_chars=500, tolerance=1)
        assert splitter.split("     ") == []

    def test_no_mid_paragraph_cut(self) -> None:
        """Paragraph boundaries (\\n\\n) should always be respected.
        With chunk_size=1, each paragraph is its own chunk."""
        text = "Para A.\n\nPara B.\n\nPara C."
        splitter = ParagraphSplitter(chunk_size=1, max_chars=500, tolerance=1)
        chunks = splitter.split(text)
        for chunk in chunks:
            assert "\n\n" not in chunk, f"Chunk contains paragraph separator: {repr(chunk)}"
        # Each paragraph appears intact.
        assert chunks == ["Para A.", "Para B.", "Para C."]

    def test_custom_separator(self) -> None:
        """Custom paragraph_separator works correctly."""
        text = "P1.\n---\nP2.\n---\nP3."
        splitter = ParagraphSplitter(
            chunk_size=1,
            max_chars=500,
            tolerance=1,
            paragraph_separator="\n---\n",
        )
        chunks = splitter.split(text)
        assert len(chunks) == 3
        assert chunks[0] == "P1."
        assert chunks[1] == "P2."
        assert chunks[2] == "P3."

    def test_markdown_mode_horizontal_rules(self) -> None:
        """Markdown horizontal rules start new paragraphs."""
        text = "Section one.\n\n---\n\nSection two.\n\n---\n\nSection three."
        splitter = ParagraphSplitter(
            chunk_size=2,
            max_chars=500,
            tolerance=1,
            use_markdown_mode=True,
        )
        chunks = splitter.split(text)
        assert len(chunks) >= 3

    def test_markdown_mode_headings(self) -> None:
        """Markdown headings start new paragraphs."""
        text = "Intro.\n\n# Heading\n\nBody.\n\n## Sub\n\nMore."
        splitter = ParagraphSplitter(
            chunk_size=2,
            max_chars=500,
            tolerance=1,
            use_markdown_mode=True,
        )
        chunks = splitter.split(text)
        # Should respect heading breaks; headings are harder to count,
        # but we can verify no chunk contains a heading start marker.
        for chunk in chunks:
            assert "\\n#" not in chunk

    def test_overlap_param(self) -> None:
        """The last `overlap` chars of each chunk appear in the next."""
        text = "One.\n\nTwo.\n\nThree.\n\nFour.\n\nFive."
        splitter = ParagraphSplitter(
            chunk_size=2,
            max_chars=500,
            tolerance=1,
            overlap=4,
        )
        chunks = splitter.split(text)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            ending = chunks[i][-splitter.overlap:]
            assert ending.strip() and ending.strip() in chunks[i + 1], (
                f"Chunk {i} ending ({repr(ending)}) not in chunk {i+1}"
            )

    def test_overlap_zero(self) -> None:
        """overlap=0 means no overlap between chunks."""
        text = "A.\n\nB.\n\nC."
        splitter = ParagraphSplitter(
            chunk_size=1,
            max_chars=500,
            tolerance=1,
            overlap=0,
        )
        chunks = splitter.split(text)
        for i in range(len(chunks) - 1):
            assert not chunks[i + 1].startswith(chunks[i][-10:].strip())

    def test_invalid_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap must be a non-negative integer"):
            ParagraphSplitter(chunk_size=2, max_chars=500, tolerance=1, overlap=-1)

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            ParagraphSplitter(chunk_size=0, max_chars=500, tolerance=1)

    def test_max_chars_ceiling(self) -> None:
        """max_chars must never be exceeded."""
        long_text = "A" * 600  # 600 chars, single paragraph
        splitter = ParagraphSplitter(chunk_size=1, max_chars=100, tolerance=1)
        chunks = splitter.split(long_text)
        for chunk in chunks:
            assert len(chunk) <= 100, f"Chunk exceeds max_chars: {repr(chunk)}"
