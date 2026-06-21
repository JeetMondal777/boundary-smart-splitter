from __future__ import annotations

import sys
import threading
from unittest.mock import patch
import pytest
import warnings

from boundary_smart_splitter.base import BaseSplitter
from boundary_smart_splitter.v1_word import WordSplitter
from boundary_smart_splitter.v2_sentence import SentenceSplitter
from boundary_smart_splitter.v3_paragraph import ParagraphSplitter
from boundary_smart_splitter.v4_structure import StructureSplitter as OriginalStructureSplitter

class StructureSplitter(OriginalStructureSplitter):
    def __init__(self, *args, **kwargs):
        if "min_tokens" not in kwargs:
            kwargs["min_tokens"] = 0
        super().__init__(*args, **kwargs)

from boundary_smart_splitter.heading_detector import (
    MarkdownHeadingDetector,
    HTMLHeadingDetector,
    PlainTextHeadingDetector,
    select_detector,
    HeadingResult,
)


class DummySplitter(BaseSplitter):
    def split(self, text: str) -> list[str]:
        return [text]

    def split_stream(self, text: str):
        yield text



def test_length_function_resolution() -> None:
    # 1. string tiktoken encoding name
    splitter = DummySplitter(chunk_size=1, tolerance=0, length_function="cl100k_base")
    assert splitter._len_fn("hello world") > 0

    # 2. string tiktoken model name
    splitter_model = DummySplitter(chunk_size=1, tolerance=0, length_function="gpt-4")
    assert splitter_model._len_fn("hello world") > 0

    # 3. custom callable
    custom_len = lambda t: len(t.split())
    splitter_custom = DummySplitter(chunk_size=1, tolerance=0, length_function=custom_len)
    assert splitter_custom._len_fn("hello world") == 2

    # 4. invalid string (mocked to simulate tiktoken installed)
    from unittest.mock import MagicMock
    mock_tiktoken = MagicMock()
    mock_tiktoken.get_encoding.side_effect = KeyError("invalid")
    mock_tiktoken.encoding_for_model.side_effect = KeyError("invalid")
    
    with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
        with pytest.raises(ValueError, match="Unsupported encoding or model name"):
            DummySplitter(chunk_size=1, tolerance=0, length_function="invalid_encoding_xyz")

    # 5. invalid type
    with pytest.raises(TypeError, match="length_function must be a string or a callable"):
        DummySplitter(chunk_size=1, tolerance=0, length_function=123)  # type: ignore


def test_tiktoken_missing_fallback() -> None:
    with patch.dict(sys.modules, {"tiktoken": None}):
        with pytest.warns(ImportWarning, match="tiktoken is not installed"):
            splitter = DummySplitter(chunk_size=1, tolerance=0, length_function="cl100k_base")
        # Should fall back to character counting
        assert splitter._len_fn("hello") == 5


def test_dual_limit_constraints() -> None:
    # WordSplitter where max_tokens triggers the limit first
    # "hello" is 1 token. "hello hello hello hello hello" is 5 tokens.
    # Char length is 29 chars.
    # Let's set max_tokens=3 (should split before reaching max_chars=100)
    text = "hello hello hello hello hello hello hello hello hello hello"
    splitter_token = WordSplitter(chunk_size=10, max_chars=100, max_tokens=3, tolerance=0)
    chunks = splitter_token.split(text)
    # Each chunk should contain at most 3 tokens (which is 3 words here)
    for chunk in chunks:
        assert len(chunk.split()) <= 3

    # WordSplitter where max_chars triggers the limit first
    # Let's set max_chars=11 (which holds at most 2 words like "hello hello")
    # while max_tokens is 5.
    splitter_char = WordSplitter(chunk_size=10, max_chars=11, max_tokens=5, tolerance=0)
    chunks_char = splitter_char.split(text)
    for chunk in chunks_char:
        assert len(chunk) <= 11
        assert len(chunk.split()) <= 2


def test_heading_detectors() -> None:
    # 1. Markdown Heading Detector
    md_detector = MarkdownHeadingDetector()
    assert md_detector.detect("# Heading 1") == HeadingResult(level=1, text="Heading 1")
    assert md_detector.detect("## Subheading") == HeadingResult(level=2, text="Subheading")
    assert md_detector.detect("Not a heading") is None

    # 2. HTML Heading Detector
    html_detector = HTMLHeadingDetector()
    assert html_detector.detect("<h1>Title</h1>") == HeadingResult(level=1, text="Title")
    assert html_detector.detect("<h2 class=\"title\">Sub</h2>") == HeadingResult(level=2, text="Sub")
    assert html_detector.detect("<div role=\"heading\" aria-level=\"3\">Aria Heading</div>") == HeadingResult(level=3, text="Aria Heading")
    assert html_detector.detect("<div>Not a heading</div>") is None

    # 3. Plain Text Heading Detector
    plain_detector = PlainTextHeadingDetector()
    # ALL CAPS
    assert plain_detector.detect("SECTION ONE") == HeadingResult(level=1, text="SECTION ONE")
    # Numbered
    assert plain_detector.detect("1. Introduction") == HeadingResult(level=1, text="Introduction")
    assert plain_detector.detect("2.3 Methods") == HeadingResult(level=2, text="Methods")
    # Short Title Case
    assert plain_detector.detect("Conclusion") == HeadingResult(level=2, text="Conclusion")
    # Too long
    assert plain_detector.detect("This line is too long to be a heading and should be ignored even if Capitalized") is None
    # Ends in punctuation
    assert plain_detector.detect("This is a sentence.") is None

    # 4. Auto select detector
    assert isinstance(select_detector("<h1>Title</h1>\n<p>text</p>"), HTMLHeadingDetector)
    assert isinstance(select_detector("# Markdown\ncontent"), MarkdownHeadingDetector)
    assert isinstance(select_detector("Plain text here\nno formatting"), PlainTextHeadingDetector)


def test_structure_splitter_heading_detector() -> None:
    # Test html detector in StructureSplitter
    html_text = "<h1>Intro</h1>\nSome text.\n<h2>Details</h2>\nMore details."
    splitter = StructureSplitter(chunk_size=1, max_chars=1500, heading_detector="html")
    chunks = splitter.split(html_text)
    assert len(chunks) == 2
    assert chunks[0].startswith("<h1>Intro</h1>")
    assert chunks[1].startswith("<h2>Details</h2>")


def test_thread_safety() -> None:
    splitter = StructureSplitter(chunk_size=1, max_chars=100, max_tokens=10)
    text = "# Title\n" + "word " * 50

    results: list[list[str]] = [[] for _ in range(10)]

    def worker(idx: int) -> None:
        results[idx] = splitter.split(text)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads must produce the exact same split outputs
    first_result = results[0]
    assert len(first_result) > 0
    for r in results[1:]:
        assert r == first_result
