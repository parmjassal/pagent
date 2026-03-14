from typing import Dict, Any, List
from .state import AgentState
from .semantic_search import SemanticSearchEngine
from .workspace import WorkspaceContext

class SemanticSearchAgent:
    """
    System Agent that manages semantic indexing and retrieval for local folders.
    """

    def __init__(self, workspace: WorkspaceContext):
        self.workspace = workspace

    def index_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node to build/refresh an index for a folder."""
        # 1. Resolve folder from metadata or prompt
        folder_to_index = state.get("metadata", {}).get("target_folder")
        if not folder_to_index:
            return {"messages": [{"role": "system", "content": "SearchAgent: No folder specified for indexing."}]}

        # 2. Initialize Engine at Session Level
        session_path = self.workspace.get_session_dir(state["user_id"], state["session_id"])
        index_dir = session_path / "semantic_index"
        engine = SemanticSearchEngine(index_dir)

        # 3. Perform Indexing
        engine.build_index(Path(folder_to_index))

        return {
            "messages": [{"role": "system", "content": f"SearchAgent: Index built for {folder_to_index}"}],
            "metadata": {"index_ready": True}
        }

    def query_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node to perform a semantic query."""
        query_text = state.get("metadata", {}).get("search_query")
        if not query_text:
            return {"messages": [{"role": "system", "content": "SearchAgent: No query provided."}]}

        session_path = self.workspace.get_session_dir(state["user_id"], state["session_id"])
        index_dir = session_path / "semantic_index"
        engine = SemanticSearchEngine.load(index_dir)

        results = engine.query(query_text)
        
        # Format results for the agent history
        formatted_results = "\n".join([f"- {r['metadata']['path']} (score: {r['score']:.2f})" for r in results])
        
        return {
            "messages": [{"role": "assistant", "content": f"Semantic Search Results:\n{formatted_results}"}],
            "search_results": results
        }
from pathlib import Path
