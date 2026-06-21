from __future__ import annotations

import pytest

from boundary_smart_splitter.v4_structure import StructureSplitter
from boundary_smart_splitter.base import Chunk, Section


def test_sibling_merging() -> None:
    # Sibling sub-sections should be merged to fill min_tokens=50.
    # Top-level heading (# Top Level) starts its own chunk and is never merged across.
    text = (
        "# Top Level\n"
        "Preamble text.\n"
        "## Sub 1\n"
        "First sibling content.\n"
        "## Sub 2\n"
        "Second sibling content.\n"
        "## Sub 3\n"
        "Third sibling content."
    )
    # min_tokens is 50, which is larger than any single Sub section
    splitter = StructureSplitter(min_tokens=50, max_tokens=300, respect_headings=True)
    chunks = splitter.split_with_metadata(text)
    
    # We expect:
    # 1. Preamble + # Top Level
    # 2. Sub 1 + Sub 2 + Sub 3 merged together (since they are siblings sharing path ('Top Level',))
    assert len(chunks) == 2
    assert "Top Level" in chunks[0].text
    assert "Sub 1" in chunks[1].text
    assert "Sub 2" in chunks[1].text
    assert "Sub 3" in chunks[1].text
    assert chunks[1].metadata["boundary_type"] == "merged_sibling"


def test_orphan_strategy() -> None:
    text = (
        "© 2025 Copyright Notice\n"
        "# Introduction\n"
        "Welcome to the project."
    )
    
    # 1. Strategy = "drop"
    splitter_drop = StructureSplitter(orphan_strategy="drop")
    chunks_drop = splitter_drop.split(text)
    assert len(chunks_drop) == 1
    assert "Copyright" not in chunks_drop[0]

    # 2. Strategy = "merge_backward"
    # Preamble cannot merge backward since there is no previous section, so it stays as preamble/starts first chunk
    splitter_merge = StructureSplitter(orphan_strategy="merge_backward")
    chunks_merge = splitter_merge.split(text)
    assert len(chunks_merge) == 2
    assert "Copyright" in chunks_merge[0]

    # Post-heading orphan matching pattern
    text_with_tail = (
        "# Introduction\n"
        "Welcome to the project.\n"
        "© 2025 Copyright Notice"
    )
    chunks_tail = splitter_merge.split(text_with_tail)
    # The copyright notice matches the orphan pattern and should be merged backward into the introduction section
    assert len(chunks_tail) == 1
    assert "Copyright Notice" in chunks_tail[0]


def test_section_aware_overlap() -> None:
    text = (
        "# Section Alpha\n"
        "First sentence of Alpha. Second sentence of Alpha.\n"
        "# Section Beta\n"
        "First sentence of Beta."
    )
    splitter = StructureSplitter(overlap_tokens=30, overlap_mode="heading")
    chunks = splitter.split_with_metadata(text)
    assert len(chunks) == 2
    # Second chunk should contain the context prefix of Alpha
    beta_chunk = chunks[1].text
    assert "[context: Section Alpha — First sentence of Alpha.]" in beta_chunk
    assert "Section Beta" in beta_chunk


def test_rich_chunk_metadata() -> None:
    text = (
        "# Title\n"
        "Content text."
    )
    splitter = StructureSplitter()
    chunks = splitter.split_with_metadata(text)
    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert meta["section_title"] == "Title"
    assert meta["heading_path"] == ["Title"]
    assert meta["heading_level"] == 1
    assert "token_count" in meta
    assert "char_count" in meta
    assert meta["position_in_doc"] == "start"
    assert meta["section_index"] == 0


def test_configurable_fallback_chain() -> None:
    text = "part1|part2|part3|part4"
    # Set max_chars=8, which forces splitting.
    splitter = StructureSplitter(max_chars=8, fallback_separators=["|"])
    chunks = splitter.split(text)
    # Since max_chars is 8, and separator is "|", it should split at "|"
    assert len(chunks) == 4
    assert chunks[0] == "part1"
    assert chunks[1] == "part2"
    assert chunks[2] == "part3"
    assert chunks[3] == "part4"
