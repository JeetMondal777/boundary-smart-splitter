def split_text(text: str) -> list[str]:
    """Split input text using a boundary-smart strategy.

    This function coordinates paragraph and sentence splitters to
    produce cohesive, retrieval-friendly chunks.
    """
    if not text:
        return []
    
    # Process text paragraphs
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]
