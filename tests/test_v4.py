"""Unit tests for the V4 StructureSplitter (section/topic-count semantics)."""

import pytest

from boundary_smart_splitter.v4_structure import StructureSplitter as OriginalStructureSplitter

class StructureSplitter(OriginalStructureSplitter):
    def __init__(self, *args, **kwargs):
        if "min_tokens" not in kwargs:
            kwargs["min_tokens"] = 0
        super().__init__(*args, **kwargs)


class TestStructureSplitter:
    def test_section_count_split(self) -> None:
        """chunk_size=1 means 1 section per chunk when boundaries exist."""
        text = (
            "# Intro\n\nContent.\n\n"
            "# Section 2\n\nMore.\n\n"
            "# Section 3\n\nEven more."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 3

    def test_respect_headings_true(self) -> None:
        """Headings act as hard section boundaries."""
        text = (
            "Preamble.\n\n"
            "# Alpha\n\nContent A.\n\n"
            "## Beta\n\nContent B."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 3
        assert chunks[0] == "Preamble."
        assert chunks[1].startswith("# Alpha")
        assert chunks[2].startswith("## Beta")

    def test_respect_headings_false(self) -> None:
        """respect_headings=False means headings do NOT split sections."""
        text = (
            "# Title\n\nBody.\n\n"
            "## Subheading\n\nMore body."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        # No boundaries at all → one section spanning the whole text.
        assert len(chunks) == 1

    def test_transition_phrases_split(self) -> None:
        """Transition phrases create section boundaries."""
        text = (
            "First topic content here.\n\n"
            "However, the second topic starts here.\n\n"
            "In summary, the third wraps up."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 3
        assert "First topic" in chunks[0]
        assert "However" in chunks[1]
        assert "In summary" in chunks[2]

    def test_transition_phrases_false(self) -> None:
        """split_on_transitions=False disables transition boundaries."""
        text = (
            "First topic.\n\n"
            "However, it continues without a split.\n\n"
            "Still the same topic."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 1

    def test_custom_transition_phrases(self) -> None:
        """Custom transition_phrases list works."""
        text = (
            "Topic one.\n\n"
            "Moving right along, here is topic two.\n\n"
            "Topic three."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            split_on_transitions=True,
            transition_phrases=["Moving right along,"],
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 2
        assert "Topic one." in chunks[0]
        assert "Moving right along" in chunks[1]

    def test_custom_transition_phrases_empty_list(self) -> None:
        """Empty custom list means no transition-based splitting."""
        text = (
            "Topic one.\n\n"
            "However, this should not split.\n\n"
            "Still together."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            split_on_transitions=True,
            transition_phrases=[],
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 1

    def test_double_newline_boundary(self) -> None:
        """Triple+ newlines create section boundaries."""
        text = (
            "Paragraph one.\n\n\n"
            "Paragraph two.\n\n\n\n"
            "Paragraph three."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            split_on_transitions=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 3
        assert "Paragraph one." in chunks[0]
        assert "Paragraph two." in chunks[1]
        assert "Paragraph three." in chunks[2]

    def test_double_newline_as_boundary_false(self) -> None:
        """double_newline_as_boundary=False disables blank-line splitting."""
        text = (
            "Paragraph one.\n\n\n"
            "This is still same section."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 1

    def test_max_chars_ceiling(self) -> None:
        """max_chars must never be exceeded."""
        text = "# H1\n\n" + ("A" * 2000) + "\n\n# H2\n\n" + ("B" * 2000)
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=100,
            tolerance=0,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        for chunk in chunks:
            assert len(chunk) <= 100, f"Chunk exceeds max_chars: {repr(chunk)}"

    def test_single_long_section_falls_back_to_v3(self) -> None:
        """A section exceeding max_chars triggers V3 paragraph splitting."""
        text = (
            "# Big Section\n\n" +
            ("This is a sentence. " * 40).strip() +
            "\n\nMore content."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=100,
            tolerance=0,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        # Should produce multiple chunks from the V3 fallback.
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 100, f"Chunk exceeds max_chars: {repr(chunk)}"

    def test_empty_string(self) -> None:
        splitter = StructureSplitter()
        assert splitter.split("") == []

    def test_whitespace_only(self) -> None:
        splitter = StructureSplitter()
        assert splitter.split("     ") == []

    def test_no_boundaries(self) -> None:
        """Text with no structural boundaries → single chunk."""
        text = "Just a plain paragraph with no headings or anything else."
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_headings_at_start(self) -> None:
        """Heading at position 0 correctly starts the first section."""
        text = "# Only Heading\n\nSome content."
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0].startswith("# Only Heading")

    def test_heading_all_levels(self) -> None:
        """All heading levels (# through ######) are detected."""
        text_parts = []
        for level in range(1, 7):
            text_parts.append(f"{'#' * level} Heading {level}\n\nBody {level}.")
        text = "\n\n".join(text_parts)
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 6

    def test_chunk_size_multiple_sections(self) -> None:
        """chunk_size=2 groups two sections per chunk."""
        text = (
            "# One\n\nA.\n\n"
            "# Two\n\nB.\n\n"
            "# Three\n\nC."
        )
        splitter = StructureSplitter(
            chunk_size=2,
            max_chars=1500,
            tolerance=0,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 2  # 3 sections → 2+1
        assert "One" in chunks[0] and "Two" in chunks[0]
        assert "Three" in chunks[1]

    def test_transition_mid_sentence_not_split(self) -> None:
        """Transition phrases mid-sentence are NOT treated as boundaries."""
        text = (
            "This is a sentence. However, it continues without splitting."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        # "However," appears mid-sentence, not after \n\n — no split.
        assert len(chunks) == 1

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            StructureSplitter(chunk_size=0, max_chars=500, tolerance=0)

    def test_invalid_max_chars(self) -> None:
        with pytest.raises(ValueError, match="max_chars must be a positive integer"):
            StructureSplitter(chunk_size=1, max_chars=0, tolerance=0)

    def test_invalid_tolerance(self) -> None:
        with pytest.raises(ValueError, match="tolerance must be a non-negative integer"):
            StructureSplitter(chunk_size=1, max_chars=500, tolerance=-1)

    def test_invalid_transition_phrases_type(self) -> None:
        with pytest.raises(TypeError):
            StructureSplitter(transition_phrases=42)  # type: ignore[arg-type]

    def test_invalid_transition_phrases_string(self) -> None:
        with pytest.raises(ValueError, match="transition_phrases must be"):
            StructureSplitter(transition_phrases="non-default")

    def test_heading_not_matched_without_space(self) -> None:
        """#WithoutSpace is not a valid Markdown heading and should not split."""
        text = (
            "#NotAHeading\n\n"
            "Still same section.\n\n"
            "#AlsoNotAHeading\n\nBody."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        # No headings detected, no other boundaries → single chunk.
        assert len(chunks) == 1

    # ── numbered sections ───────────────────────────────────────────

    def test_numbered_sections_default(self) -> None:
        """Numbered sections (1., 2., 3.) create boundaries by default."""
        text = (
            "Intro.\n\n"
            "1. Home Page\n\nContent about home.\n\n"
            "2. About Us\n\nContent about us.\n\n"
            "3. Services\n\nContent about services."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 4  # intro + 3 numbered sections
        assert chunks[0] == "Intro."
        assert chunks[1].startswith("1. Home Page")
        assert chunks[2].startswith("2. About Us")
        assert chunks[3].startswith("3. Services")

    def test_numbered_sections_alone_on_line(self) -> None:
        """A lone '1.' on its own line still starts a section."""
        text = (
            "Preamble.\n\n"
            "1.\n\nServices Page\n\nContent.\n\n"
            "2.\n\nProjects Page\n\nMore content."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 3
        assert chunks[0] == "Preamble."
        assert "1." in chunks[1]
        assert "Services Page" in chunks[1]
        assert "2." in chunks[2]
        assert "Projects Page" in chunks[2]

    def test_numbered_sections_false(self) -> None:
        """respect_numbered_sections=False disables numbered-section detection."""
        text = (
            "Intro.\n\n"
            "1. Home Page\n\nContent.\n\n"
            "2. About Us\n\nContent."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            respect_numbered_sections=False,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) == 1  # no boundaries → one chunk

    def test_numbered_sections_mid_text_ignored(self) -> None:
        """Numbered pattern mid-line (not at line start) is NOT a boundary."""
        text = (
            "Introduction.\n\n"
            "See section 1. for details and section 2. for more.\n\n"
            "Conclusion."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        # "1." and "2." appear mid-sentence, not at line start — no split.
        assert len(chunks) == 1

    def test_numbered_sections_decimal_not_matched(self) -> None:
        """Decimal numbers (3.14) are NOT treated as section boundaries."""
        text = (
            "Math chapter.\n\n"
            "The value of pi is 3.14 approximately.\n\n"
            "More text."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            split_on_transitions=False,
            double_newline_as_boundary=False,
        )
        chunks = splitter.split(text)
        # No valid numbered section markers → one chunk.
        assert len(chunks) == 1

    def test_numbered_sections_with_pdf_noise(self) -> None:
        """Numbered sections survive PDF whitespace noise."""
        text = (
            "Intro.\n\n\n"
            "1. Home Page\n\nContent.\n\n\n"
            "2. About Us\n\nContent."
        )
        splitter = StructureSplitter(
            chunk_size=1,
            max_chars=1500,
            tolerance=0,
            respect_headings=False,
            split_on_transitions=False,
        )
        chunks = splitter.split(text)
        assert len(chunks) >= 2
        # Numbered section titles should NOT be separated from their numbers.
        for chunk in chunks:
            if chunk.startswith("1."):
                assert "Home Page" in chunk
            if chunk.startswith("2."):
                assert "About Us" in chunk
