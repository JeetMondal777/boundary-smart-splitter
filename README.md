# boundary-smart-splitter

> A **boundary-first, size-second** text splitter for Python.
> LangChain-compatible. Framework-agnostic core. No ML dependencies.

```
Boundary-smart-splitter respects your text structure.
Word → Sentence → Paragraph → Section/Topic
```

Most text splitters hit a character count and slice — regardless of whether they cut through a sentence, a paragraph, or a heading. This library flips that: **chunk_size is measured in semantic units** (words, sentences, paragraphs, sections), and `max_chars` acts only as a hard safety ceiling to protect your embedding model's context window.

---

## Installation

```bash
pip install boundary-smart-splitter
```

For LangChain support, install the optional dependency:

```bash
pip install boundary-smart-splitter[langchain]
```

**No ML dependencies required.** Every splitter below is deterministic, regex-based, and works 100% offline.

---

## Design Philosophy

| Principle | How it works |
|---|---|
| **Boundary-first, size-second** | Clean boundaries are more important than exact character counts |
| **Semantic units** | `chunk_size` means words for V1, sentences for V2, paragraphs for V3, sections for V4 |
| **max_chars as safety net** | Never exceeds your embedding model's context window, but only as a last resort |
| **Graceful fallback chain** | V4 → V3 → V2 → V1 → hard char-cut |
| **LangChain-compatible, not dependent** | The core never imports LangChain; wrappers are optional |

---

## Quick Start

### V1 — Word-Boundary Splitter

Never cuts a word in half. `chunk_size` = number of words.

```python
from boundary_smart_splitter import WordSplitter

text = "The quick brown fox jumps over the lazy dog. " * 100
splitter = WordSplitter(chunk_size=60, max_chars=500, tolerance=10)
chunks = splitter.split(text)

for chunk in chunks:
    print(chunk)
```

---

### V2 — Sentence-Boundary Splitter

Never cuts a sentence in half. Finds the nearest `.`, `?`, or `!` within tolerance.

```python
from boundary_smart_splitter import SentenceSplitter

text = "Hello world. How are you? I am fine! " * 100
splitter = SentenceSplitter(chunk_size=5, max_chars=500, tolerance=2)
chunks = splitter.split(text)

for chunk in chunks:
    print(chunk)
```

#### Custom Abbreviations

Prevent mid-abbreviation splits (e.g. `Dr.`, `U.S.A.`, `e.g.`):

```python
splitter = SentenceSplitter(
    chunk_size=5,
    max_chars=500,
    tolerance=2,
    abbreviations={"Dr.", "U.S.A.", "e.g.", "i.e."},
)
```

#### Boundary Preference

```python
# Forward-first (default): scan forward for sentence end, then backward
splitter = SentenceSplitter(chunk_size=5, max_chars=500, tolerance=2, boundary_preference="forward")

# Backward-first: scan backward first, then forward
splitter = SentenceSplitter(chunk_size=5, max_chars=500, tolerance=2, boundary_preference="backward")
```

---

### V3 — Paragraph-Boundary Splitter

Never breaks a paragraph (`\n\n`). `chunk_size` = number of paragraphs.

```python
from boundary_smart_splitter import ParagraphSplitter

text = "Para one.\n\nPara two.\n\nPara three."
splitter = ParagraphSplitter(chunk_size=2, max_chars=500, tolerance=1)
chunks = splitter.split(text)

for chunk in chunks:
    print(chunk)
```

#### Markdown Mode

Respects horizontal rules and headings as extra paragraph boundaries:

```python
splitter = ParagraphSplitter(
    chunk_size=2,
    max_chars=500,
    tolerance=1,
    use_markdown_mode=True,
)
```

---

### V4 — Structure & Topic-Aware Splitter

Respects headings, numbered sections, and transition phrases. `chunk_size` = number of sections.

```python
from boundary_smart_splitter import StructureSplitter

text = """
# Intro
Welcome to our proposal.

1. Home Page
The homepage will act as a strong first impression...

2. About Us Page
To build credibility and trust.

However, we must also consider costs.

3. Services Page
A detailed breakdown of all services.
"""

splitter = StructureSplitter(
    chunk_size=1,
    max_chars=1500,
    respect_headings=True,
    respect_numbered_sections=True,
    split_on_transitions=True,
)
chunks = splitter.split(text)

for chunk in chunks:
    print("---")
    print(chunk)
```

#### V4 Configuration

| Parameter | Default | Description |
|---|---|---|
| `respect_headings` | `True` | Treat `# Heading` as a hard boundary |
| `respect_numbered_sections` | `True` | Treat `1. Title` as a boundary |
| `split_on_transitions` | `True` | Treat "However," / "In summary," etc. as boundaries |
| `transition_phrases` | built-in list | Custom list of transition phrases |
| `double_newline_as_boundary` | `True` | Treat `\n\n\n` as a boundary |

---

## Fallback Chain

When a single unit exceeds `max_chars`, the library gracefully falls back to the next-smaller boundary:

```
StructureSplitter (V4)
  → section too big? → ParagraphSplitter (V3)
    → paragraph too big? → SentenceSplitter (V2)
      → sentence too big? → WordSplitter (V1)
        → word too big? → hard char-cut
```

This ensures you **never** get a chunk larger than `max_chars`, but you **always** get the cleanest possible boundary.

---

## LangChain Integration

All splitters have LangChain-compatible wrappers. The core never imports LangChain — wrappers are optional.

```python
from boundary_smart_splitter.langchain import (
    LangChainWordSplitter,
    LangChainSentenceSplitter,
    LangChainParagraphSplitter,
    LangChainStructureSplitter,
)

splitter = LangChainStructureSplitter(chunk_size=1, max_chars=1500)
chunks = splitter.split_text("Your long document here...")
```

---

## API Comparison

| Splitter | `chunk_size` unit | `max_chars` role | Fallback | Key Params |
|---|---|---|---|---|
| `WordSplitter` | words | hard ceiling, never exceeded | hard char-cut | `tolerance` |
| `SentenceSplitter` | sentences | if sentence exceeds it, fall back to V1 | `WordSplitter` | `tolerance`, `boundary_preference`, `abbreviations` |
| `ParagraphSplitter` | paragraphs | if paragraph exceeds it, fall back to V2 | `SentenceSplitter` | `tolerance`, `overlap`, `paragraph_separator`, `use_markdown_mode` |
| `StructureSplitter` | sections / topics | if section exceeds it, fall back to V3 | `ParagraphSplitter` | `respect_headings`, `respect_numbered_sections`, `split_on_transitions` |

---

## Why boundary-first matters

**Size-first** splitting (the common approach):
```
"...the quick brown fox jum" | "ps over the lazy dog..."
# Bad: cuts through a word
```

**Boundary-first** splitting (this library):
```
"...the quick brown fox" | "jumps over the lazy dog..."
# Good: clean word boundary
```

This becomes critical at sentence level:
```
"Please visit the U.S.A. for travel. Yes!"
# sentence-aware: kept together
# naive char-cut: "U.S.A." might get split
```

And at paragraph/section level for RAG:
```
"1. Home Page\n\nThe homepage..." | "2. About Us\n\nTo build..."
# section-aware: one chunk per page/section
# naive: cuts through the middle of a page description
```

---

## Features

- **Boundary-first, size-second**: clean boundaries always take priority over exact counts
- **Semantic units for `chunk_size`**: words count words, sentences count sentences, paragraphs count paragraphs, sections count sections
- **`max_chars` as universal safety net**: protects embedding model limits without breaking the semantic contract
- **Graceful fallback chain**: V4 → V3 → V2 → V1, each level catches what the level above can't handle
- **LangChain-compatible, not dependent**: the core never imports LangChain
- **Each version is a superset**: V2 includes V1, V3 includes出门 include V2, V4 includes V3
- **No ML dependencies**: fast, deterministic, offline-capable
- **Abbreviation-aware**: configurable abbreviation list prevents false sentence breaks
- **Markdown-aware**: optional Markdown paragraph/heading detection
- **PDF-friendly**: handles leading whitespace from PDF text extraction

---

## Requirements

- Python >= 3.9

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

MIT © 2026
