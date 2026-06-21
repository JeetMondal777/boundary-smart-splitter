import pytest
from boundary_smart_splitter.v4_structure import StructureSplitter


def test_quality_scoring_ideal_chunk():
    """An ideal chunk with no issues should score 1.0 and have no quality flags."""
    # We want a chunk that:
    # - lands on a clean 'soft' boundary (score 0.5)
    # - token count is within [min_tokens, target_max] (score 0.5)
    # - has no orphan_merged or tail_merged (deductions 0.0)
    text = "# Introduction\n" + "word " * 220  # ~220 tokens
    splitter = StructureSplitter(
        chunk_size=1,
        min_tokens=100,
        max_tokens=500,
    )
    chunks = splitter.split_with_metadata(text)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    
    assert chunk.metadata["boundary_type"] == "soft"
    assert chunk.metadata["quality_flags"] == []
    assert chunk.metadata["quality_score"] == 1.0


def test_quality_scoring_under_min_tokens():
    """An undersized chunk should have 'under_min_tokens' flag and a reduced score."""
    text = "# Small Section\nOnly two words."
    splitter = StructureSplitter(
        chunk_size=1,
        min_tokens=200,
    )
    chunks = splitter.split_with_metadata(text)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    
    assert "under_min_tokens" in chunk.metadata["quality_flags"]
    # Score should be boundary (0.5) + token consistency (0.5 * token_count / 200) - deductions
    assert chunk.metadata["quality_score"] < 1.0
    assert chunk.metadata["quality_score"] > 0.5


def test_quality_scoring_orphan_merged():
    """A chunk with merged orphan content should have 'orphan_merged' flag and 0.1 deduction."""
    text = "# Main Section\nNormal content here.\n\n\n© 2026 Copyright"
    splitter = StructureSplitter(
        chunk_size=1,
        min_tokens=0,  # avoid under_min_tokens flag
        orphan_strategy="merge_backward",
    )
    chunks = splitter.split_with_metadata(text)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    
    assert "orphan_merged" in chunk.metadata["quality_flags"]
    # Score should be boundary (0.5) + token (0.5) - deduction (0.1) = 0.9
    assert abs(chunk.metadata["quality_score"] - 0.9) < 1e-5


def test_quality_scoring_forced_fallback():
    """A chunk forced to split via fallback should have 'forced_fallback' flag and boundary score of 0.1."""
    text = "# Section\n" + "word " * 100
    splitter = StructureSplitter(
        chunk_size=1,
        min_tokens=0,
        max_tokens=10,  # very small limit to force fallback
    )
    chunks = splitter.split_with_metadata(text)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert "forced_fallback" in chunk.metadata["quality_flags"]
        # Since it is forced fallback (hard boundary), boundary score is 0.1
        # Token score should be 0.5 (since size is <= 10)
        # Deductions: 0.0
        # Expected score: 0.6
        assert abs(chunk.metadata["quality_score"] - 0.6) < 1e-5


def test_quality_scoring_tail_merged():
    """A chunk that had a tail merged into it should have 'tail_merged' flag and 0.05 deduction."""
    # Sub 1 is long enough (>= 30 tokens), Sub 2 is short (< 30 tokens).
    # Since Sub 1 >= 30, they won't merge in sibling merging, but Sub 2 will merge backward in tail merging.
    text = "# Parent\n## Sub 1\n" + "word " * 40 + "\n## Sub 2\nShort."
    splitter = StructureSplitter(
        chunk_size=1,
        min_tokens=30,
    )
    chunks = splitter.split_with_metadata(text)
    
    # 2 chunks: Chunk 1 is '# Parent', Chunk 2 is '## Sub 1 ... ## Sub 2'
    assert len(chunks) == 2
    chunk = chunks[1]
    
    assert "tail_merged" in chunk.metadata["quality_flags"]
    # Should have a 0.05 deduction
    assert chunk.metadata["quality_score"] < 1.0
