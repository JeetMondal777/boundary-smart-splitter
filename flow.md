# FLOW — boundary-smart-splitter

> Development log & architecture walkthrough.

---

## Project Structure

```
boundary-smart-splitter/
├── boundary_smart_splitter/
│   ├── __init__.py              ← Package entry, exports all splitters
│   ├── base.py                  ← BaseSplitter ABC (shared by all versions)
│   ├── v1_word.py               ← V1: Word-Boundary Splitter (word-count)
│   ├── v2_sentence.py           ← V2: Sentence-Boundary Splitter (sentence-count)
│   ├── v3_paragraph.py          ← V3: Paragraph-Boundary Splitter (paragraph-count)
│   └── langchain/
│       ├── __init__.py          ← Exports LangChain wrappers
│       └── compat.py            ← LangChain wrappers (optional dependency)
├── tests/
│   ├── test_v1.py               ← 11 tests for WordSplitter
│   ├── test_v2.py               ← 13 tests for SentenceSplitter
│   ├── test_v3.py               ← 15 tests for ParagraphSplitter
│   └── fixtures/                ← (empty, reserved for future test data)
├── examples/
│   └── basic_usage.py           ← V1 demo script
├── pyproject.toml               ← Package metadata, deps, tool config
├── README.md                    ← User-facing docs
├── LICENSE                      ← MIT
├── .gitignore                   ← Python + IDE + OS
└── Roadmap.md                   ← Full feature roadmap
```

---

## Design Philosophy — Semantic Units

Every splitter measures `chunk_size` in its **own semantic unit**.
A `max_chars` ceiling acts as a universal safety net.

| Splitter | `chunk_size` unit | Default | `max_chars` Default |
|---|---|---|---|
| `WordSplitter` | words | 60 | 500 |
| `SentenceSplitter` | sentences | 5 | 500 |
| `ParagraphSplitter` | paragraphs | 2 | 500 |

**Fallback chain:** ParagraphSplitter → SentenceSplitter → WordSplitter → hard char-cut

---

## BaseSplitter (`base.py`)

```
BaseSplitter
├── chunk_size: int     — target count of semantic units
├── max_chars: int      — hard character ceiling (never exceeded)
├── tolerance: int      — how far to scan for clean boundaries
├── split(text) -> List[str]  ← abstract
└── _strip_and_filter(chunks) -> List[str]
```

---

## V1 — WordSplitter (`v1_word.py`)

**Semantic unit:** Words (runs of non-whitespace characters)

### Algorithm

```
1. Tokenise text into words → [(start, end), ...]
2. Group words into chunks of `chunk_size` words each
3. If a group exceeds `max_chars` characters:
     → Drop trailing words until it fits
     → If even 1 word exceeds max_chars → hard-cut at max_chars boundaries
4. Strip and return each chunk
```

### Key decisions

- **Word = non-whitespace run.** Token boundaries are natural split points.
- **`max_chars` overrides `chunk_size`.** Safety takes priority.
- **Hard char-cut fallback.** A 500-char word with `max_chars=50` → 10 pieces of 50 chars. This preserves the universal ceiling invariant.

---

## V2 — SentenceSplitter (`v2_sentence.py`)

**Semantic unit:** Sentences (delimited by `.`, `?`, `!`)

### Sentence tokenisation

Sentences end at `.`, `?`, or `!` UNLESS the punctuation is part of a
known abbreviation (e.g. `U.S.A.`). The abbreviation detection checks
multi-period abbreviations by finding every period position and verifying
whether it falls inside any known abbreviation span.

Default abbreviations list includes: `e.g.`, `i.e.`, `etc.`, `Dr.`, `U.S.A.`,
month abbreviations (`Jan.`, `Feb.`, ...), and more. Overridable via param.

### Algorithm

```
1. Tokenise text into sentences → [(start, end), ...]
2. Group sentences into chunks of `chunk_size` sentences each
3. If a group exceeds `max_chars`:
     → Drop trailing sentences until it fits
     → If even 1 sentence exceeds max_chars:
         → Fall back to V1 (word-split) for that sentence
4. If no sentence boundaries found at all:
     → Fall back to V1 directly (handles fragmented or unpunctuated text)
```

---

## V3 — ParagraphSplitter (`v3_paragraph.py`)

**Semantic unit:** Paragraphs (delimited by `\n\n`)

### Paragraph tokenisation

Paragraphs are delimited by `\n\n` (configurable via `paragraph_separator`).
When `use_markdown_mode=True`, horizontal rules (`---`, `***`) and
headings (`#`, `##`, etc.) also act as paragraph boundaries.

### Algorithm

```
1. Tokenise text into paragraphs → [(start, end), ...]
2. Group paragraphs into chunks of `chunk_size` paragraphs each
3. If a group exceeds `max_chars`:
     → Drop trailing paragraphs until it fits
     → If even 1 paragraph exceeds max_chars:
         → Fall back to V2 (sentence-split) for that paragraph,
           which may fall further back to V1
4. If overlap > 0: prepend the last `overlap` chars of chunk N
   to the start of chunk N+1 for context continuity
5. Strip and return each chunk
```

---

## Data Flow — Fallback Chain

```
Input text
    │
    ▼
┌─────────────────────────┐
│  ParagraphSplitter.split │
│  (counts paragraphs)     │
└────────────┬────────────┘
             │ paragraph exceeds max_chars?
             ├── NO  → emit paragraph group
             └── YES → fall back to ▼
                        ┌─────────────────────────┐
                        │  SentenceSplitter.split  │
                        │  (counts sentences)      │
                        └────────────┬────────────┘
                                     │ sentence exceeds max_chars?
                                     ├── NO  → emit sentence group
                                     └── YES → fall back to ▼
                                                ┌──────────────────────────┐
                                                │  WordSplitter.split      │
                                                │  (counts words)          │
                                                └────────────┬─────────────┘
                                                             │ word exceeds max_chars?
                                                             ├── NO  → emit word group
                                                             └── YES → HARD CHAR-CUT
                                                                      (split at max_chars)
```

---

## Test Coverage

| Test Suite | Tests | Status |
|---|---|---|
| test_v1.py (WordSplitter) | 11 | ✅ |
| test_v2.py (SentenceSplitter) | 13 | ✅ |
| test_v3.py (ParagraphSplitter) | 15 | ✅ |
| **Total** | **39** | **✅ All passing** |

---

## LangChain Wrappers

Both splitters have LangChain-compatible wrappers:

```python
from boundary_smart_splitter.langchain import (
    LangChainWordSplitter,
    LangChainSentenceSplitter,
)

# V1
splitter = LangChainWordSplitter(chunk_size=60, max_chars=500)
chunks = splitter.split_text(text)

# V2
splitter = LangChainSentenceSplitter(chunk_size=5, max_chars=500)
chunks = splitter.split_text(text)
```

V3 LangChain wrapper can be added when needed.
