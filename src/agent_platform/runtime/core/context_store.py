import logging
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..orch.state import AgentRole

logger = logging.getLogger(__name__)

class ContextStore(ABC):
    """
    Interface for hierarchical global_context management.
    Enforces lexical scoping: children see parent context, but not vice versa.
    """
    @abstractmethod
    def list_facts(self, agent_id: str) -> List[str]:
        """Lists all fact IDs visible to this agent (including ancestors)."""
        pass

    @abstractmethod
    def read_fact(self, agent_id: str, fact_id: str) -> Optional[str]:
        """Reads a specific fact content."""
        pass

    @abstractmethod
    def update_fact(self, agent_id: str, fact_id: str, content: str, role: AgentRole):
        """Allows supervisors to commit a fact to their own context."""
        pass

class FilesystemContextStore(ContextStore):
    """
    Hierarchical filesystem implementation. 
    Performs recursive upward lookup at query-time.
    """
    def __init__(self, session_root: Path):
        self.session_root = session_root
        self.agents_root = session_root / "agents"

    def _get_agent_dir(self, agent_id: str) -> Path:
        return self.agents_root / agent_id

    def _get_ancestor_contexts(self, agent_id: str) -> List[Path]:
        """
        Walks up the directory tree from the agent directory to the session root,
        collecting all 'global_context' directories found.
        """
        agent_dir = self._get_agent_dir(agent_id)
        visible_paths = []
        
        current = agent_dir
        # Traverse up from the agent's specific directory until we reach the session root
        while current and current != self.session_root:
            context_path = current / "global_context"
            if context_path.exists() and context_path.is_dir():
                visible_paths.append(context_path)
            
            # Move to parent
            if current == self.agents_root:
                break
            current = current.parent
            
        # Finally, check for a global context at the session root itself
        root_context = self.session_root / "global_context"
        if root_context.exists() and root_context.is_dir():
            visible_paths.append(root_context)
            
        return visible_paths

    def list_facts(self, agent_id: str) -> List[str]:
        facts = set()
        for context_dir in self._get_ancestor_contexts(agent_id):
            for fact_file in context_dir.glob("*.md"):
                facts.add(fact_file.stem)
        return list(facts)

    def read_fact(self, agent_id: str, fact_id: str) -> Optional[str]:
        for context_dir in self._get_ancestor_contexts(agent_id):
            path = context_dir / f"{fact_id}.md"
            if path.exists():
                return path.read_text()
        return None

    def update_fact(self, agent_id: str, fact_id: str, content: str, role: AgentRole):
        if role != AgentRole.SUPERVISOR:
            raise PermissionError(f"Agent {agent_id} with role {role} is not authorized to update context.")
        
        agent_dir = self._get_agent_dir(agent_id)
        context_dir = agent_dir / "global_context"
        context_dir.mkdir(parents=True, exist_ok=True)
        
        path = context_dir / f"{fact_id}.md"
        path.write_text(content)
        logger.info(f"Fact '{fact_id}' updated by Supervisor {agent_id}")
