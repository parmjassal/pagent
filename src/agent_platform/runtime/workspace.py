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
        return self.get_session_dir(user_id, session_id) / "agents" / agent_id

    def ensure_session_structure(self, user_id: str, session_id: str):
        """Creates the basic directory structure for a session."""
        session_path = self.get_session_dir(user_id, session_id)
        (session_path / "agents").mkdir(parents=True, exist_ok=True)
        (session_path / "skills").mkdir(parents=True, exist_ok=True)
        (session_path / "prompts").mkdir(parents=True, exist_ok=True)
        (session_path / "snapshot").mkdir(parents=True, exist_ok=True)
        return session_path
