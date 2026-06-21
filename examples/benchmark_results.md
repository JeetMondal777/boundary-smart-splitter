# Boundary-Smart-Splitter Benchmark Report

This report evaluates splitters on diverse document types across multiple chunk-quality metrics.

## Document: Wikipedia Article (Markdown)

| Metric | StructureSplitter | RecursiveCharacterTextSplitter (recursive) | CharacterTextSplitter (character) |
|:---| :--- | :--- | :--- |
| Avg tokens/chunk | 81.0 | 162.0 | 162.0 |
| CV (consistency) | 0.89 | 0.00 | 0.00 |
| Chunks < 100 tokens | 50.0% | 0.0% | 0.0% |
| Chunks > 512 tokens | 0.0% | 0.0% | 0.0% |
| Cross-topic mixing | 50.0% | 100.0% | 100.0% |
| Heading preservation | 40.0% | 20.0% | 20.0% |
| Processing time | 27.9ms | 0.2ms | 0.0ms |

## Document: Academic Paper (Dirty Text)

| Metric | StructureSplitter | RecursiveCharacterTextSplitter (recursive) | CharacterTextSplitter (character) |
|:---| :--- | :--- | :--- |
| Avg tokens/chunk | 26.4 | 132.0 | 132.0 |
| CV (consistency) | 0.56 | 0.00 | 0.00 |
| Chunks < 100 tokens | 100.0% | 0.0% | 0.0% |
| Chunks > 512 tokens | 0.0% | 0.0% | 0.0% |
| Cross-topic mixing | 0.0% | 100.0% | 100.0% |
| Heading preservation | 100.0% | 0.0% | 0.0% |
| Processing time | 1.0ms | 0.0ms | 0.0ms |

## Document: News Article (HTML)

| Metric | StructureSplitter | RecursiveCharacterTextSplitter (recursive) | CharacterTextSplitter (character) |
|:---| :--- | :--- | :--- |
| Avg tokens/chunk | 51.0 | 102.0 | 102.0 |
| CV (consistency) | 0.14 | 0.00 | 0.00 |
| Chunks < 100 tokens | 100.0% | 0.0% | 0.0% |
| Chunks > 512 tokens | 0.0% | 0.0% | 0.0% |
| Cross-topic mixing | 50.0% | 100.0% | 100.0% |
| Heading preservation | 66.7% | 33.3% | 33.3% |
| Processing time | 1.0ms | 0.0ms | 0.0ms |

## Document: Python Code with Docstrings

| Metric | StructureSplitter | RecursiveCharacterTextSplitter (recursive) | CharacterTextSplitter (character) |
|:---| :--- | :--- | :--- |
| Avg tokens/chunk | 82.0 | 82.0 | 82.0 |
| CV (consistency) | 0.00 | 0.00 | 0.00 |
| Chunks < 100 tokens | 100.0% | 100.0% | 100.0% |
| Chunks > 512 tokens | 0.0% | 0.0% | 0.0% |
| Cross-topic mixing | 0.0% | 0.0% | 0.0% |
| Heading preservation | 100.0% | 100.0% | 100.0% |
| Processing time | 0.5ms | 0.0ms | 0.0ms |
