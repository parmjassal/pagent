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

    def _get_agent_dir(self, agent_id: str) -> Path:
        return self.session_root / "agents" / agent_id

    def _get_ancestor_contexts(self, agent_id: str) -> List[Path]:
        """
        Determines the list of visible global_context directories.
        In this skeleton, we'll implement a simple one-level parent lookup.
        In production, this would use the session's agent hierarchy metadata.
        """
        agent_dir = self._get_agent_dir(agent_id)
        visible_paths = []
        
        # 1. Current Agent Context
        if (agent_dir / "global_context").exists():
            visible_paths.append(agent_dir / "global_context")
            
        # 2. Session Root Context (Optional common base)
        if (self.session_root / "global_context").exists():
            visible_paths.append(self.session_root / "global_context")
            
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
