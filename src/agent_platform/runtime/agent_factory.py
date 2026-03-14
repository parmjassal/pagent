from pathlib import Path
from typing import Dict, Any, Optional
from .workspace import WorkspaceContext
from .state import AgentState, create_initial_state
from .quota import SessionQuota

class AgentFactory:
    """Handles the creation of new agents and their filesystem context."""

    def __init__(self, workspace: WorkspaceContext):
        self.workspace = workspace

    def create_agent(
        self, 
        user_id: str, 
        session_id: str, 
        agent_id: str,
        current_quota: SessionQuota,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentState]:
        """
        Creates a new agent's directory structure and returns its initial state.
        Returns None if quota limits are exceeded.
        """
        if not current_quota.can_spawn():
            return None

        # 1. Resolve paths
        agent_dir = self.workspace.get_agent_dir(user_id, session_id, agent_id)
        inbox_path = agent_dir / "inbox"
        outbox_path = agent_dir / "outbox"

        # 2. Ensure directories exist
        inbox_path.mkdir(parents=True, exist_ok=True)
        outbox_path.mkdir(parents=True, exist_ok=True)

        # 3. Initialize state
        # Note: We pass the current max_agents to the new state's quota
        state = create_initial_state(
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            inbox_path=inbox_path,
            outbox_path=outbox_path,
            max_agents=current_quota.max_agents
        )
        
        # 4. Increment agent count (This will be added to the state via the reducer)
        state["quota"].agent_count = 1 
        
        return state
