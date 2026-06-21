import pytest
from typing import Generator
from boundary_smart_splitter.base import BoundarySmartSplitterError
from boundary_smart_splitter.v1_word import WordSplitter
from boundary_smart_splitter.v2_sentence import SentenceSplitter
from boundary_smart_splitter.v3_paragraph import ParagraphSplitter
from boundary_smart_splitter.v4_structure import StructureSplitter, Chunk
from boundary_smart_splitter.langchain.compat import (
    LangChainWordSplitter,
    LangChainSentenceSplitter,
    LangChainParagraphSplitter,
    LangChainStructureSplitter,
)


def test_split_stream_generator():
    """Verify that split_stream actually returns a Generator and yields lazily."""
    text = "Word1 Word2 Word3 Word4 Word5"
    splitter = WordSplitter(chunk_size=2)
    stream = splitter.split_stream(text)
    
    # Assert it is a generator
    assert isinstance(stream, Generator)
    
    # Get first chunk and check
    first = next(stream)
    assert first == "Word1 Word2"
    
    # Get remaining
    remaining = list(stream)
    assert remaining == ["Word3 Word4", "Word5"]


def test_split_with_metadata_stream():
    """Verify that split_with_metadata_stream is a generator."""
    text = "# Section 1\nPara 1\n# Section 2\nPara 2"
    splitter = StructureSplitter(chunk_size=1, min_tokens=0)
    stream = splitter.split_with_metadata_stream(text)
    
    assert isinstance(stream, Generator)
    first = next(stream)
    assert isinstance(first, Chunk)
    assert "Section 1" in first.text
    
    remaining = list(stream)
    assert len(remaining) == 1
    assert isinstance(remaining[0], Chunk)


def test_diagnostics_validation_errors():
    """Verify that BoundarySmartSplitterError is raised for invalid inputs."""
    with pytest.raises(BoundarySmartSplitterError, match="chunk_size must be a positive integer"):
        WordSplitter(chunk_size=-1)
        
    with pytest.raises(BoundarySmartSplitterError, match="max_chars must be a positive integer"):
        WordSplitter(max_chars=0)
        
    with pytest.raises(BoundarySmartSplitterError, match="max_tokens must be a positive integer"):
        WordSplitter(max_tokens=-5)

    with pytest.raises(BoundarySmartSplitterError, match="tolerance must be a non-negative integer"):
        WordSplitter(tolerance=-2)


def test_langchain_wrappers_params_and_mapping():
    """Verify LangChain wrappers map parameters properly and accept advanced arguments."""
    # 1. LangChainWordSplitter
    # chunk_size <= 10 -> word limit
    lc_word = LangChainWordSplitter(chunk_size=5, tolerance=2)
    assert lc_word._splitter.chunk_size == 5
    assert lc_word._splitter.tolerance == 2
    
    # chunk_size > 10 -> character limit mapping
    lc_word_mapped = LangChainWordSplitter(chunk_size=100)
    assert lc_word_mapped._splitter.chunk_size == 60  # default_chunk_size
    assert lc_word_mapped._splitter.max_chars == 100
    
    # 2. LangChainSentenceSplitter
    lc_sent = LangChainSentenceSplitter(chunk_size=3, boundary_preference="backward")
    assert lc_sent._splitter.chunk_size == 3
    assert lc_sent._splitter.boundary_preference == "backward"
    
    lc_sent_mapped = LangChainSentenceSplitter(chunk_size=250)
    assert lc_sent_mapped._splitter.chunk_size == 5
    assert lc_sent_mapped._splitter.max_chars == 250
    
    # 3. LangChainParagraphSplitter
    lc_para = LangChainParagraphSplitter(chunk_size=1, overlap=10, paragraph_separator="\n\n")
    assert lc_para._splitter.chunk_size == 1
    assert lc_para._splitter.overlap == 10
    
    # Mapped with chunk_overlap
    lc_para_mapped = LangChainParagraphSplitter(chunk_size=120, chunk_overlap=15)
    assert lc_para_mapped._splitter.chunk_size == 2
    assert lc_para_mapped._splitter.max_chars == 120
    assert lc_para_mapped._splitter.overlap == 15
    
    # 4. LangChainStructureSplitter
    lc_struct = LangChainStructureSplitter(
        chunk_size=2,
        min_tokens=50,
        respect_headings=False,
        overlap_mode="tail",
        overlap_tokens=20,
    )
    assert lc_struct._splitter.chunk_size == 2
    assert lc_struct._splitter.min_tokens == 50
    assert lc_struct._splitter.respect_headings is False
    assert lc_struct._splitter.overlap_mode == "tail"
    assert lc_struct._splitter.overlap_tokens == 20
    
    # Mapped with chunk_overlap
    lc_struct_mapped = LangChainStructureSplitter(chunk_size=800, chunk_overlap=40)
    assert lc_struct_mapped._splitter.chunk_size == 1
    assert lc_struct_mapped._splitter.max_chars == 800
    assert lc_struct_mapped._splitter.overlap_tokens == 40
