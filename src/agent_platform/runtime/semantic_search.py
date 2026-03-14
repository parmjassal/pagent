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
    Sparse (SPLADE-like) and LSH approach.
    """

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.index_path / "metadata.json"
        self.index_file = self.index_path / "index.json"
        
        # In-memory storage for the skeleton
        self.documents: Dict[str, str] = {} # doc_id -> content
        self.vector_index: Dict[str, List[float]] = {} # doc_id -> sparse_vector

    def build_index(self, folder_path: Path, glob_pattern: str = "**/*.*"):
        """Recursively finds files and builds the semantic index."""
        logger.info(f"Building semantic index for: {folder_path}")
        
        for file_path in folder_path.glob(glob_pattern):
            if file_path.is_file() and not self._should_ignore(file_path):
                try:
                    content = file_path.read_text(errors='ignore')
                    doc_id = hashlib.sha256(str(file_path).encode()).hexdigest()
                    
                    # 1. Generate SPLADE-like sparse vector (Placeholder)
                    # In production: vector = splade_model.encode(content)
                    vector = self._generate_simulated_splade(content)
                    
                    self.documents[doc_id] = str(file_path)
                    self.vector_index[doc_id] = vector
                except Exception as e:
                    logger.error(f"Failed to index {file_path}: {e}")
        
        self._save_index()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the index and returns ranked results."""
        # 1. Encode Query (Placeholder)
        query_vec = self._generate_simulated_splade(query_text)
        
        # 2. LSH / Similiarity Search (Placeholder)
        # For the skeleton, we'll do a simple dot-product on the "sparse" vectors
        results = []
        for doc_id, doc_vec in self.vector_index.items():
            score = sum(a * b for a, b in zip(query_vec, doc_vec))
            results.append({"doc_id": doc_id, "path": self.documents[doc_id], "score": score})
        
        # Rank and return
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _should_ignore(self, path: Path) -> bool:
        ignore_list = [".git", "__pycache__", ".venv", ".DS_Store", "node_modules"]
        return any(part in path.parts for part in ignore_list)

    def _generate_simulated_splade(self, text: str) -> List[float]:
        """Simulates a sparse lexical expansion vector with basic term weighting."""
        vec = [0.0] * 512 # Increase dimensionality to 512
        words = text.lower().split()
        for w in words:
            # Simple length-based weighting to simulate 'importance'
            weight = len(w) * 0.1
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % 512
            vec[idx] += weight
        return vec

    def _save_index(self):
        data = {
            "documents": self.documents,
            "vectors": self.vector_index
        }
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
