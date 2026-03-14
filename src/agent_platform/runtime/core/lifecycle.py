import shutil
import logging
from pathlib import Path
from typing import Optional, List
from .workspace import WorkspaceContext
from .agent_factory import AgentFactory
from ..orch.state import AgentState
from ..orch.quota import SessionQuota

logger = logging.getLogger(__name__)

class AgentLifecycleManager:
    """
    Manages the lifecycle of agents within a session.
    Handles high-level creation, cleanup, and archival of agent filesystem pointers.
    """

    def __init__(self, workspace: WorkspaceContext, factory: AgentFactory):
        self.workspace = workspace
        self.factory = factory

    def create_agent(
        self, 
        user_id: str, 
        session_id: str, 
        agent_id: str, 
        current_quota: SessionQuota,
        parent_depth: int = 0
    ) -> Optional[AgentState]:
        """Orchestrates agent creation through the factory."""
        logger.info(f"Creating agent lifecycle context for {agent_id} in session {session_id}")
        return self.factory.create_agent(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            current_quota=current_quota,
            parent_depth=parent_depth
        )

    def cleanup_agent(self, user_id: str, session_id: str, agent_id: str):
        """Irreversibly removes an agent's workspace (inbox/outbox)."""
        agent_dir = self.workspace.get_agent_dir(user_id, session_id, agent_id)
        if agent_dir.exists():
            logger.info(f"Cleaning up agent directory: {agent_dir}")
            shutil.rmtree(agent_dir)

    def archive_agent(self, user_id: str, session_id: str, agent_id: str):
        """Moves agent data to an archive directory within the session."""
        session_dir = self.workspace.get_session_dir(user_id, session_id)
        archive_root = session_dir / "archive" / "agents" / agent_id
        archive_root.mkdir(parents=True, exist_ok=True)

        agent_dir = self.workspace.get_agent_dir(user_id, session_id, agent_id)
        if agent_dir.exists():
            logger.info(f"Archiving agent {agent_id} to {archive_root}")
            for item in agent_dir.iterdir():
                shutil.move(str(item), str(archive_root / item.name))
            agent_dir.rmdir()

    def list_active_agents(self, user_id: str, session_id: str) -> List[str]:
        """Lists IDs of agents currently having an active workspace in the session."""
        agents_root = self.workspace.get_session_dir(user_id, session_id) / "agents"
        if not agents_root.exists():
            return []
        return [d.name for d in agents_root.iterdir() if d.is_dir()]
