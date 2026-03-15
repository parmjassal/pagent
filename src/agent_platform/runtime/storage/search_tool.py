from typing import Dict, Any, List, Optional
from pathlib import Path
from ..storage.semantic_search import SemanticSearchEngine
from ..orch.state import AgentState

class SearchTools:
    """
    Tools for semantic indexing and searching of local repositories.
    """
    def __init__(self, session_path: Path):
        self.session_path = session_path
        self.index_dir = session_path / "semantic_index"

    def build_index(self, path: str, state: Optional[AgentState] = None) -> str:
        """
        Builds a semantic index for the specified directory path.
        """
        target_path = Path(path)
        if not target_path.exists():
            return f"Error: Path {path} does not exist."
        
        engine = SemanticSearchEngine(self.index_dir)
        engine.build_index(target_path)
        return f"Successfully built semantic index for {path}"

    def semantic_search(self, query: str, state: Optional[AgentState] = None) -> str:
        """
        Performs a semantic query against the built index.
        """
        if not self.index_dir.exists():
            return "Error: No semantic index found. Call build_index first."
        
        engine = SemanticSearchEngine.load(self.index_dir)
        results = engine.query(query)
        
        if not results:
            return "No matching documents found."
            
        formatted = "\n".join([
            f"### {r['metadata']['path']} (Score: {r['score']:.2f})\n{r['metadata'].get('snippet', '')}"
            for r in results
        ])
        return formatted
