from typing import List, Dict, Any, Optional
from ..core.context_store import ContextStore
from ..orch.state import AgentState

class ContextTools:
    """
    Tools for agents to interact with the hierarchical global_context.
    """
    def __init__(self, store: ContextStore):
        self.store = store

    def list_context(self, state: AgentState) -> List[str]:
        """Lists all fact IDs visible to the current agent."""
        return self.store.list_facts(state["agent_id"])

    def read_context(self, state: AgentState, fact_id: str) -> Optional[str]:
        """Reads a specific fact from the hierarchy."""
        return self.store.read_fact(state["agent_id"], fact_id)

    def update_context(self, state: AgentState, fact_id: str, content: str) -> str:
        """Allows supervisors to add or update a fact in their own context."""
        try:
            self.store.update_fact(state["agent_id"], fact_id, content, state["role"])
            return f"Successfully updated fact '{fact_id}' in global_context."
        except PermissionError as e:
            return str(e)

    def search_context(self, state: AgentState, query: str) -> str:
        """
        Performs a semantic-ish search across visible global_context files.
        (Currently implements simple keyword matching across the hierarchy).
        """
        visible_facts = self.store.list_facts(state["agent_id"])
        results = []
        for fid in visible_facts:
            content = self.store.read_fact(state["agent_id"], fid)
            if query.lower() in content.lower():
                results.append(f"### {fid}\n{content[:200]}...")
        
        if not results:
            return "No matching context found."
        return "\n\n".join(results)
