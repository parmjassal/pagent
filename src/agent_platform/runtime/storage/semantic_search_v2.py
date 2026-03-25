import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """
    Handles indexing and querying of local files using a hybrid 
    Sparse and LSH approach. Supports targeted file indexing and keyword boosting.
    """

    def __init__(self, index_path: Path, chunk_size: int = 100):
        self.index_path = index_path
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index_file_path = self.index_path / "index.json"
        self.chunk_size = chunk_size  # Lines per chunk
        
        self.documents: Dict[str, Dict[str, Any]] = {}  # doc_id -> metadata
        self.vector_index: Dict[str, List[float]] = {}  # doc_id -> vector

    def index_chunk(
        self,
        file_path: str,
        content: str,
        start_line: int,
        boost_keywords: Optional[List[str]] = None
    ):
        """
        Indexes a single chunk directly. This allows external/manual chunk indexing.
        """
        try:
            # Estimate end_line from content
            line_count = content.count("\n") + 1
            end_line = start_line + line_count - 1

            # Stable unique ID
            doc_id = hashlib.sha256(
                f"{file_path}:{start_line}:{end_line}".encode()
            ).hexdigest()

            vector = self._lexical_expand(content, boost_keywords=boost_keywords)

            self.documents[doc_id] = {
                "path": str(file_path),
                "start_line": start_line,
                "end_line": end_line,
                "content": content
            }

            self.vector_index[doc_id] = vector

        except Exception as e:
            logger.error(f"Failed to index chunk for {file_path}: {e}")

    def index_file(self, file_path: Path, keywords: Optional[List[str]] = None):
        """Indexes a specific file with optional domain-specific keyword boosting."""
        if not file_path.is_file() or self._should_ignore(file_path):
            return

        try:
            lines = file_path.read_text(errors='ignore').splitlines()
            
            for i in range(0, len(lines), self.chunk_size):
                chunk_lines = lines[i : i + self.chunk_size]
                content = "\n".join(chunk_lines)
                
                start_line = i + 1
                end_line = i + len(chunk_lines)
                
                # Unique ID for this specific file chunk
                doc_id = hashlib.sha256(
                    f"{file_path.absolute()}:{start_line}:{end_line}".encode()
                ).hexdigest()
                
                # Expand lexical vector with optional keyword boosting
                vector = self._lexical_expand(content, boost_keywords=keywords)
                
                self.documents[doc_id] = {
                    "path": str(file_path.absolute()),
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": content  # Stored for retrieval snippets
                }
                self.vector_index[doc_id] = vector
            
            # Save state after each file is successfully processed
            self._save_index()
            
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")

    def build_index(self, folder_path: Path, glob_pattern: str = "**/*.*"):
        """Recursively finds files and builds the index (Bulk Mode)."""
        logger.info(f"Building bulk semantic index for: {folder_path}")
        for file_path in folder_path.glob(glob_pattern):
            self.index_file(file_path)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the index and returns ranked chunks."""
        query_vec = self._lexical_expand(query_text)
        results = []
        
        for doc_id, doc_vec in self.vector_index.items():
            # Dot product for similarity
            score = sum(a * b for a, b in zip(query_vec, doc_vec))
            if score > 0:
                results.append({
                    "doc_id": doc_id, 
                    "metadata": self.documents[doc_id], 
                    "score": score
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def list_unique_files(self) -> List[str]:
        """Returns a unique list of file paths present in this index."""
        unique_paths = {doc["path"] for doc in self.documents.values()}
        return sorted(list(unique_paths))

    def _lexical_expand(self, text: str, boost_keywords: Optional[List[str]] = None) -> List[float]:
        """Creates a 512-dim vector with specific boosts for architectural keywords."""
        vec = [0.0] * 512 
        words = text.lower().split()
        
        # Pre-process boost keywords for O(1) lookup
        boost_set = {
            word
            for k in boost_keywords or []
            for word in k.lower().split()
        }

        for w in words:
            # Base weight by length
            weight = len(w) * 0.1
            
            # Significant multiplier if the word is a domain-specific "Important Keyword"
            if w in boost_set:
                weight *= 5.0 
            
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % 512
            vec[idx] += weight
        return vec

    def _should_ignore(self, path: Path) -> bool:
        ignore_list = [".git", "__pycache__", ".venv", ".vcli", ".DS_Store", "node_modules", "output"]
        return any(part in path.parts for part in ignore_list)

    def _save_index(self):
        data = {"documents": self.documents, "vectors": self.vector_index}
        with open(self.index_file_path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, index_path: Path) -> 'SemanticSearchEngine':
        engine = cls(index_path)
        if engine.index_file_path.exists():
            try:
                with open(engine.index_file_path, "r") as f:
                    data = json.load(f)
                    engine.documents = data.get("documents", {})
                    engine.vector_index = data.get("vectors", {})
            except Exception as e:
                logger.error(f"Failed to load index file: {e}")
        return engine