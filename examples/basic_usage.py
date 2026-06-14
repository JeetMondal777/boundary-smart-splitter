"""Examples: basic usage of all four boundary-smart-splitters.

Each splitter measures ``chunk_size`` in its own semantic unit:
  - WordSplitter        → chunk_size = number of **words**
  - SentenceSplitter    → chunk_size = number of **sentences**
  - ParagraphSplitter   → chunk_size = number of **paragraphs**
  - StructureSplitter   → chunk_size = number of **sections/topics**

``max_chars`` is always a hard character ceiling.
"""

from boundary_smart_splitter import (
    WordSplitter,
    SentenceSplitter,
    ParagraphSplitter,
    StructureSplitter,
)

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "The five boxing wizards jump quickly. "
    "Sphinx of black quartz, judge my vow. "
    "Two driven jocks help fax my big quiz. "
) * 2


def demo_word_splitter() -> None:
    """Demonstrate V1: word-count-based splitting.

    ``chunk_size=20`` → each chunk contains ~20 words (semantic unit).
    """
    splitter = WordSplitter(chunk_size=20, max_chars=500, tolerance=5)
    chunks = splitter.split(TEXT)

    print("=== WordSplitter (chunk_size=20 words) ===")
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks, 1):
        wc = len(chunk.split())
        print(f"  Chunk {i} ({wc} words, {len(chunk)} chars): {chunk[:80]}...")


def demo_sentence_splitter() -> None:
    """Demonstrate V2: sentence-count-based splitting.

    ``chunk_size=2`` → each chunk contains ~2 sentences.
    Falls back to word-splitting for long sentences.
    """
    splitter = SentenceSplitter(chunk_size=2, max_chars=500, tolerance=1)
    chunks = splitter.split(TEXT)

    print("\n=== SentenceSplitter (chunk_size=2 sentences) ===")
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks, 1):
        sc = chunk.count(".") + chunk.count("!") + chunk.count("?")
        print(f"  Chunk {i} (~{sc} sentences, {len(chunk)} chars): {chunk[:80]}...")


def demo_paragraph_splitter() -> None:
    """Demonstrate V3: paragraph-count-based splitting.

    ``chunk_size=2`` → each chunk contains ~2 paragraphs.
    Supports ``overlap`` for context continuity and ``use_markdown_mode``
    for Markdown heading/horizontal-rule boundaries.
    """
    para_text = "\n\n".join(
        [
            "Introduction paragraph here.",
            "Second paragraph with more content.",
            "Third paragraph goes here.",
            "Fourth paragraph for the demo.",
            "Fifth and final paragraph.",
        ]
    )

    splitter = ParagraphSplitter(
        chunk_size=2,
        max_chars=500,
        tolerance=1,
        overlap=15,
    )
    chunks = splitter.split(para_text)

    print("\n=== ParagraphSplitter (chunk_size=2 paragraphs, overlap=15) ===")
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks, 1):
        pc = chunk.count("\n\n") + 1
        print(f"  Chunk {i} ({pc} paragraphs, {len(chunk)} chars):")
        print(f"    {repr(chunk[:100])}...")


def demo_fallback_chain() -> None:
    """Demonstrate the V3 → V2 → V1 fallback chain.

    When a paragraph exceeds ``max_chars``, ParagraphSplitter falls back
    to SentenceSplitter; when a sentence exceeds ``max_chars``, it falls
    further to WordSplitter; when a single word still exceeds ``max_chars``,
    a hard character-cut is applied.
    """
    # A single paragraph with no line breaks, way over max_chars
    long_text = "Hello world. " + "x" * 600 + " Goodbye world."

    splitter = ParagraphSplitter(
        chunk_size=1,
        max_chars=100,
        tolerance=1,
    )
    chunks = splitter.split(long_text)

    print("\n=== Fallback Chain Demo (single long paragraph/sentence) ===")
    print(f"Total chunks: {len(chunks)} (all ≤100 chars)\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i} ({len(chunk)} chars): {repr(chunk[:80])}...")


def demo_structure_splitter() -> None:
    """Demonstrate V4: section/topic-aware splitting.

    ``chunk_size=1`` → each section (heading group) is its own chunk.
    Falls back to V3 paragraph splitting within long sections.
    """
    md_text = (
        "# Introduction\n\n"
        "Opening paragraph here.\n\n"
        "# Core Concepts\n\n"
        "First concept.\n\n"
        "However, there is a related idea.\n\n"
        "# Conclusion\n\n"
        "Wrapping up with final thoughts."
    )

    splitter = StructureSplitter(
        chunk_size=1,
        max_chars=1500,
        tolerance=0,
        respect_headings=True,
        split_on_transitions=True,
        double_newline_as_boundary=True,
    )
    chunks = splitter.split(md_text)

    print("\n=== StructureSplitter (chunk_size=1 section) ===")
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks, 1):
        sec_type = "heading" if chunk.startswith("#") else "transition" if any(
            p in chunk for p in ["However", "In summary"]
        ) else "content"
        print(f"  Chunk {i} ({sec_type}, {len(chunk)} chars):")
        print(f"    {repr(chunk[:90])}...")


if __name__ == "__main__":
    demo_word_splitter()
    demo_sentence_splitter()
    demo_paragraph_splitter()
    demo_fallback_chain()
    demo_structure_splitter()
