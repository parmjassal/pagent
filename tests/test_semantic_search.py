import pytest
from pathlib import Path
from agent_platform.runtime.semantic_search import SemanticSearchEngine

def test_index_and_query(tmp_path):
    # 1. Create dummy files
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "apple.txt").write_text("I love eating fresh red apples.")
    (docs_dir / "banana.txt").write_text("The yellow banana is a great fruit.")
    
    index_dir = tmp_path / "index"
    engine = SemanticSearchEngine(index_dir)
    
    # 2. Build Index
    engine.build_index(docs_dir)
    
    # 3. Query
    results = engine.query("fruit banana")
    
    # Banana should rank higher for the query 'banana'
    assert len(results) == 2
    assert "banana.txt" in results[0]["metadata"]["path"]
    assert results[0]["score"] > results[1]["score"]

def test_persistence(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.txt").write_text("persistence test")
    
    index_dir = tmp_path / "index"
    engine = SemanticSearchEngine(index_dir)
    engine.build_index(docs_dir)
    
    # Load from disk
    new_engine = SemanticSearchEngine.load(index_dir)
    assert len(new_engine.documents) == 1
    doc_metadata = list(new_engine.documents.values())[0]
    assert "test.txt" in doc_metadata["path"]
