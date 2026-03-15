from pathlib import Path
from typing import Dict, Any, Optional
from .workspace import WorkspaceContext
from ..orch.state import AgentState, create_initial_state, AgentRole
from ..orch.quota import SessionQuota

class AgentFactory:
    """Handles the creation of new agents and their filesystem context."""

    def __init__(self, workspace: WorkspaceContext, max_spawn_depth: int = 5):
        self.workspace = workspace
        self.max_spawn_depth = max_spawn_depth

    def get_agent_db_path(self, user_id: str, session_id: str, agent_id: str) -> Path:
        """Returns the path to the agent's SQLite state database."""
        return self.workspace.get_agent_dir(user_id, session_id, agent_id) / "state.db"

    def create_agent(
        self, 
        user_id: str, 
        session_id: str, 
        agent_id: str,
        current_quota: SessionQuota,
        parent_depth: int = 0,
        generated_output: Optional[str] = None,
        role: AgentRole = AgentRole.WORKER, 
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentState]:
        """
        Creates a new agent's directory structure and returns its initial state.
        """
        if not current_quota.can_spawn() or parent_depth >= self.max_spawn_depth:
            return None

        # 1. Resolve paths
        agent_dir = self.workspace.get_agent_dir(user_id, session_id, agent_id)
        inbox_path = agent_dir / "inbox"
        outbox_path = agent_dir / "outbox"
        knowledge_path = self.workspace.get_session_dir(user_id, session_id) / "knowledge"

        # 2. Ensure directories exist
        inbox_path.mkdir(parents=True, exist_ok=True)
        outbox_path.mkdir(parents=True, exist_ok=True)

        # 3. Initialize state
        state = create_initial_state(
            agent_id=agent_id,
            role=role, 
            user_id=user_id,
            session_id=session_id,
            inbox_path=inbox_path,
            outbox_path=outbox_path,
            knowledge_path=knowledge_path,
            current_depth=parent_depth + 1,
            max_agents=current_quota.max_agents,
            generated_output=generated_output
        )
        
        # 4. Increment agent count
        state["quota"].agent_count = 1 
        
        return state
