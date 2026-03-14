import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """
    Handles indexing and querying of local files using a hybrid 
    Sparse and LSH approach. Supports chunking for big files.
    """

    def __init__(self, index_path: Path, chunk_size: int = 100):
        self.index_path = index_path
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_path / "index.json"
        self.chunk_size = chunk_size # Lines per chunk
        
        self.documents: Dict[str, Dict[str, Any]] = {} # doc_id -> metadata
        self.vector_index: Dict[str, List[float]] = {} # doc_id -> vector

    def build_index(self, folder_path: Path, glob_pattern: str = "**/*.*"):
        """Recursively finds files and builds the semantic index with chunking."""
        logger.info(f"Building chunked semantic index for: {folder_path}")
        
        for file_path in folder_path.glob(glob_pattern):
            if file_path.is_file() and not self._should_ignore(file_path):
                try:
                    lines = file_path.read_text(errors='ignore').splitlines()
                    
                    # Create Chunks
                    for i in range(0, len(lines), self.chunk_size):
                        chunk_lines = lines[i : i + self.chunk_size]
                        content = "\n".join(chunk_lines)
                        
                        start_line = i + 1
                        end_line = i + len(chunk_lines)
                        
                        # doc_id includes line range to ensure uniqueness per chunk
                        doc_id = hashlib.sha256(f"{file_path}:{start_line}:{end_line}".encode()).hexdigest()
                        
                        vector = self._lexical_expand(content)
                        
                        self.documents[doc_id] = {
                            "path": str(file_path),
                            "start_line": start_line,
                            "end_line": end_line
                        }
                        self.vector_index[doc_id] = vector
                except Exception as e:
                    logger.error(f"Failed to index {file_path}: {e}")
        
        self._save_index()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the index and returns ranked chunks."""
        query_vec = self._lexical_expand(query_text)
        results = []
        for doc_id, doc_vec in self.vector_index.items():
            score = sum(a * b for a, b in zip(query_vec, doc_vec))
            results.append({
                "doc_id": doc_id, 
                "metadata": self.documents[doc_id], 
                "score": score
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _should_ignore(self, path: Path) -> bool:
        ignore_list = [".git", "__pycache__", ".venv", ".DS_Store", "node_modules"]
        return any(part in path.parts for part in ignore_list)

    def _lexical_expand(self, text: str) -> List[float]:
        vec = [0.0] * 512 
        words = text.lower().split()
        for w in words:
            weight = len(w) * 0.1
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % 512
            vec[idx] += weight
        return vec

    def _save_index(self):
        data = {"documents": self.documents, "vectors": self.vector_index}
        with open(self.index_file, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, index_path: Path) -> 'SemanticSearchEngine':
        engine = cls(index_path)
        if engine.index_file.exists():
            with open(engine.index_file, "r") as f:
                data = json.load(f)
                engine.documents = data.get("documents", {})
                engine.vector_index = data.get("vectors", {})
        return engine
