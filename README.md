# boundary-smart-splitter

> A **boundary-first, size-second** text splitter for Python.
> LangChain-compatible. Framework-agnostic core. No ML dependencies.

```
Boundary-smart-splitter respects your text structure.
Word → Sentence → Paragraph → Section/Topic
```

Most text splitters hit a character count and slice — regardless of whether they cut through a sentence, a paragraph, or a heading. This library flips that: **chunk_size is measured in semantic units** (words, sentences, paragraphs, sections), and `max_chars` acts only as a hard safety ceiling to protect your embedding model's context window.

## Benchmarks & Evaluation

We evaluate `StructureSplitter` against standard LangChain splitters on real-world articles. Scores (out of 100) are weighted based on Topic Coherence (40%), Sizing Safety (30%), Processing Speed (20%), and Boundary Accuracy (10%).

### 1. AI Agents Glossary

Evaluated on the [NVIDIA AI Agents Glossary Article](https://www.nvidia.com/en-us/glossary/ai-agents):

| Splitter | Overall Score (/100) | Topic Coherence | Sizing Safety | Avg Speed | Boundary Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **StructureSplitter (Ours)** | **77.3** | **62.8%** | 90.5% | 14.36ms | **10.0/10** (Clean Sentence splits) |
| **RecursiveCharacterTextSplitter** | **73.8** | 50.2% | 92.5% | **0.17ms** | 6.0/10 (Cuts mid-sentence) |
| **SemanticChunker** | **55.9** | 38.0% | 85.7% | 130.35ms | **10.0/10** (Clean Sentence splits) |
| **TokenTextSplitter** | **55.3** | 6.1% | **96.2%** | 2.59ms | 6.0/10 (Cuts mid-sentence) |

### 2. RAG Complete Guide

Evaluated on the [DataCamp RAG Complete Guide Article](https://www.datacamp.com/blog/what-is-retrieval-augmented-generation-rag):

| Splitter | Overall Score (/100) | Topic Coherence | Sizing Safety | Avg Speed | Boundary Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **RecursiveCharacterTextSplitter** | **82.3** | 65.6% | 86.8% | **0.10ms** | **10.0/10** (Paragraph aligned) |
| **StructureSplitter (Ours)** | **79.9** | **70.2%** | **89.5%** | 14.05ms | **10.0/10** (Clean Sentence splits) |
| **TokenTextSplitter** | **55.2** | 6.2% | **95.8%** | 2.20ms | 6.0/10 (Cuts mid-sentence) |
| **SemanticChunker** | **50.8** | 30.6% | 78.6% | 131.17ms | **10.0/10** (Clean Sentence splits) |

### 3. Agent Orchestration

Evaluated on the [Lyzr Agent Orchestration Article](https://www.lyzr.ai/blog/agent-orchestration):

| Splitter | Overall Score (/100) | Topic Coherence | Sizing Safety | Avg Speed | Boundary Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **StructureSplitter (Ours)** | **80.8** | **88.1%** | 85.2% | 44.63ms | **10.0/10** (Clean Sentence splits) |
| **RecursiveCharacterTextSplitter** | **62.9** | 18.5% | 98.4% | **0.43ms** | 6.0/10 (Cuts mid-sentence) |
| **SemanticChunker** | **58.6** | 47.3% | 82.1% | 289.44ms | **10.0/10** (Clean Sentence splits) |
| **TokenTextSplitter** | **52.7** | 11.6% | **100.0%** | 4.75ms | 0.0/10 (Cuts mid-word) |

---

## Installation

```bash
pip install boundary-smart-splitter
```

For LangChain support, install the optional dependency:

```bash
pip install boundary-smart-splitter[langchain]
```

### Token-Aware Splitting (Highly Recommended)

To use token limits (`max_tokens` / `min_tokens`) accurately and consistently, you should install `tiktoken`:

```bash
pip install tiktoken
```

> [!IMPORTANT]
> **Strictly Recommended for Production**: While `boundary-smart-splitter` runs out-of-the-box using character-count fallback approximation, installing `tiktoken` is **strongly urged** to prevent inconsistent split sizes. True token-aware boundaries prevent context-window overflow and guarantee optimal coherence for embedding models.

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
from tiktoken import tiktoken #Optional Dependency For token aware splitting (recommended)

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
    max_tokens=300, # only remove if you want to go with normal flow
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
| `respect_headings` | `True` | Treat detected headings as hard section boundaries |
| `respect_numbered_sections` | `True` | Treat `1. Title` as a boundary |
| `split_on_transitions` | `True` | Treat "However," / "In summary," etc. as boundaries |
| `transition_phrases` | built-in list | Custom list of transition phrases |
| `double_newline_as_boundary` | `True` | Treat `\n\n\n` as a boundary |
| `heading_detector` | `"auto"` | Heading detector strategy (`"auto"`, `"markdown"`, `"html"`, `"plain"`, or a custom `HeadingDetector` class) |
| `min_tokens` | `200` | Target minimum tokens per chunk. Smaller sibling sections are merged up to this budget |
| `orphan_strategy` | `"merge_backward"` | How to handle content with no heading (boilerplate/byline/copyright). Choices: `"merge_backward"`, `"drop"`, `"tag_only"` |
| `orphan_patterns` | built-in list | List of regex patterns to match orphan/boilerplate text |
| `overlap_mode` | `"heading"` | Style of overlap: `"heading"` (context-aware: injects parent heading + first sentence context) or `"tail"` (repeats trailing tokens) |
| `overlap_tokens` | `0` | Size of overlap in tokens (opt-in) |
| `overlap_prefix_template` | `"[context: {heading} — {first_sentence}]"` | Template used in `"heading"` overlap mode |
| `fallback_separators` | `["\n\n", "\n", ".", " "]` | Ordered separators to fall back on when a section exceeds limits |

#### Hierarchy-Aware Sibling Merging & Trailing Tails
Smaller sub-sections under the same parent heading path (siblings) are automatically merged up to the `min_tokens` budget (default `200` tokens). This prevents generating tiny chunks for brief subsections while avoiding cross-topic mixing (boundaries never cross top-level headings).

#### Rich Hierarchical Metadata
Use `split_with_metadata()` to retrieve chunk text along with structured metadata.

```python
from boundary_smart_splitter import StructureSplitter

text = """
# Top Level
Welcome to the project.
## Sub 1
First sibling content.
## Sub 2
Second sibling content.
"""

splitter = StructureSplitter(min_tokens=50)
chunks = splitter.split_with_metadata(text)

for chunk in chunks:
    print(f"Text: {chunk.text!r}")
    print(f"Metadata: {chunk.metadata}")
```

Each `Chunk` contains `text` (str) and a `metadata` (dict) containing:
* `section_title`: The heading title of the section (str)
* `heading_path`: List of headings from root to this section (list of str)
* `heading_level`: Depth of the current section heading (int)
* `token_count`: Number of tokens in the chunk (int)
* `char_count`: Character length of the chunk (int)
* `boundary_type`: How the boundary was determined (`"merged_sibling"`, `"soft"`, `"hard"`)
* `overlap_injected`: Whether context overlap was injected into the chunk (bool)
* `source_format`: Format detected (`"markdown"`, `"html"`, `"plain"`)
* `section_index`: Order of section in the output (int)
* `position_in_doc`: Position in document (`"start"`, `"middle"`, `"end"`)
#### Streaming & Generator Outputs
For memory-efficient processing of extremely large documents, all splitters support generator-based streaming via `split_stream()`. For V4 `StructureSplitter`, you can also use `split_with_metadata_stream()` to dynamically stream chunks along with metadata:

```python
from boundary_smart_splitter import StructureSplitter

splitter = StructureSplitter(chunk_size=1)

# Yields chunks dynamically as they are formed without full buffering
for chunk in splitter.split_stream(large_text):
    print(chunk)

# Yields rich Chunk objects containing text and metadata
for chunk_obj in splitter.split_with_metadata_stream(large_text):
    print(chunk_obj.text, chunk_obj.metadata)
```

#### Context-Aware Overlaps
Instead of blindly repeating trailing characters, V4 can inject parent heading and first-sentence context from the previous chunk.

```python
splitter = StructureSplitter(overlap_tokens=30, overlap_mode="heading")
```

#### Configurable Fallback Chains
Customize the order of separators when a section is oversized and needs to be split:

```python
# Fall back sequentially using custom separators
splitter = StructureSplitter(
    max_chars=1000,
    fallback_separators=["\n\n", "|", "\n", " "]
)
```

---

## Token-Aware Splitting

All splitters support token-aware splitting. While `max_chars` limits chunks by character length, you can also enforce `max_tokens` (measured using `tiktoken` or a custom callable).

```python
from boundary_smart_splitter import StructureSplitter

splitter = StructureSplitter(
    chunk_size=1,
    max_tokens=300,            # Hard token ceiling per chunk
    max_chars=1500,            # Optional secondary character ceiling
    length_function="gpt-4",   # Can be encoding name (e.g. "cl100k_base"), model name, or callable
)
```

If `tiktoken` is not installed, the library gracefully falls back to character-length counting (using `len()`) and issues an `ImportWarning`.

---

## Heading Detection Interface

V4 `StructureSplitter` supports pluggable heading detection. You can specify a string key (`"markdown"`, `"html"`, `"plain"`) or pass a custom class inheriting from `HeadingDetector`:

```python
from boundary_smart_splitter import StructureSplitter, HeadingDetector, HeadingResult

class CustomHeadingDetector(HeadingDetector):
    def detect(self, line: str, context=None):
        if line.startswith("!!!"):
            return HeadingResult(level=1, text=line.strip("! "))
        return None

splitter = StructureSplitter(heading_detector=CustomHeadingDetector())
```

---

## Fallback Chain

When a single unit exceeds `max_chars` or `max_tokens`, the library gracefully falls back to the next-smaller boundary:

```
StructureSplitter (V4)
  → section too big? → ParagraphSplitter (V3)
    → paragraph too big? → SentenceSplitter (V2)
      → sentence too big? → WordSplitter (V1)
        → word too big? → hard cut (character or token bounds)
```

This ensures you **never** exceed your model constraints, while always maintaining the cleanest possible boundaries.

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

splitter = LangChainStructureSplitter(chunk_size=1, max_chars=1500, max_tokens=300)
chunks = splitter.split_text("Your long document here...")
```

---

## API Comparison

| Splitter | `chunk_size` unit | Primary limits | Fallback | Key Params |
|---|---|---|---|---|
| `WordSplitter` | words | `max_chars`, `max_tokens` | hard cut | `tolerance`, `length_function` |
| `SentenceSplitter` | sentences | if limits exceeded, fall back to V1 | `WordSplitter` | `tolerance`, `boundary_preference`, `abbreviations`, `length_function` |
| `ParagraphSplitter` | paragraphs | if limits exceeded, fall back to V2 | `SentenceSplitter` | `tolerance`, `overlap`, `paragraph_separator`, `use_markdown_mode`, `length_function` |
| `StructureSplitter` | sections / topics | if limits exceeded, fall back to V3 | `ParagraphSplitter` | `respect_headings`, `heading_detector`, `respect_numbered_sections`, `split_on_transitions`, `length_function` |

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
- **Each version is a superset**: V2 includes V1, V3 includes V2, V4 includes V3
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
