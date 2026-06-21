from __future__ import annotations

import pytest
from boundary_smart_splitter.v4_structure import StructureSplitter
from boundary_smart_splitter.evaluator import Evaluator, EvaluationReport

def test_evaluator_basic() -> None:
    text = (
        "# Heading A\n"
        "Some text under A.\n"
        "# Heading B\n"
        "Some text under B."
    )
    
    splitter = StructureSplitter(min_tokens=0, max_chars=100)
    evaluator = Evaluator()
    report = evaluator.evaluate(text, splitter, compare_with=["recursive"])
    
    assert isinstance(report, EvaluationReport)
    data = report.to_dict()
    
    # Assert main splitter results exist
    assert "StructureSplitter" in data
    main_metrics = data["StructureSplitter"]
    
    # Validate metrics keys exist and have correct formats
    assert "avg_tokens_per_chunk" in main_metrics
    assert "token_distribution_cv" in main_metrics
    assert "pct_chunks_under_100_tokens" in main_metrics
    assert "pct_chunks_over_512_tokens" in main_metrics
    assert "cross_topic_mixing_rate" in main_metrics
    assert "heading_preservation_rate" in main_metrics
    assert "processing_time_ms" in main_metrics
    
    # Assert competitor results exist
    competitor_key = "RecursiveCharacterTextSplitter (recursive)"
    assert competitor_key in data
    
    # Validate report formatting
    summary_md = report.summary()
    assert "Metric" in summary_md
    assert "StructureSplitter" in summary_md
    assert "RecursiveCharacterTextSplitter" in summary_md
    
    # Check bounds
    assert 0.0 <= main_metrics["cross_topic_mixing_rate"] <= 1.0
    assert 0.0 <= main_metrics["heading_preservation_rate"] <= 1.0
    assert main_metrics["processing_time_ms"] >= 0.0
