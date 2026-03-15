from pathlib import Path
import os
from typing import Optional

class WorkspaceContext:
    """Manages the hierarchical .pagent workspace structure."""

    def __init__(self, root: Optional[Path] = None):
        if root is None:
            env_root = os.getenv("AGENT_WORKSPACE_ROOT")
            self.root = Path(env_root).expanduser().resolve() if env_root else Path("~/.pagent").expanduser().resolve()
        else:
            self.root = root.resolve()

    def get_global_dir(self) -> Path:
        return self.root / "global"

    def get_user_dir(self, user_id: str) -> Path:
        return self.root / f"user_{user_id}"

    def get_session_dir(self, user_id: str, session_id: str) -> Path:
        return self.get_user_dir(user_id) / session_id

    def get_agent_dir(self, user_id: str, session_id: str, agent_id: str) -> Path:
        # Hierarchy: agents/{id}
        return self.get_session_dir(user_id, session_id) / "agents" / agent_id

    def ensure_session_structure(self, user_id: str, session_id: str):
        """Creates basic session structure (Knowledge and Snapshots remain at session root)."""
        session_path = self.get_session_dir(user_id, session_id)
        (session_path / "skills").mkdir(parents=True, exist_ok=True)
        (session_path / "prompts").mkdir(parents=True, exist_ok=True)
        (session_path / "knowledge").mkdir(parents=True, exist_ok=True) # Big file analysis results
        (session_path / "snapshot").mkdir(parents=True, exist_ok=True)
        return session_path

    def ensure_agent_structure(self, user_id: str, session_id: str, agent_id: str):
        """Initializes an agent's private workspace with control and context dirs."""
        agent_dir = self.get_agent_dir(user_id, session_id, agent_id)
        (agent_dir / "inbox").mkdir(parents=True, exist_ok=True)
        (agent_dir / "outbox").mkdir(parents=True, exist_ok=True)
        (agent_dir / "todo").mkdir(parents=True, exist_ok=True)
        (agent_dir / "global_context").mkdir(parents=True, exist_ok=True) # Supervisor-writable
        return agent_dir
