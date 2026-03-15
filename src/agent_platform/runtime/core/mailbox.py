from pathlib import Path
import json
from abc import ABC, abstractmethod
from typing import Dict, Any

class MailboxProvider(ABC):
    @abstractmethod
    def send(self, agent_id: str, message: Dict[str, Any]):
        """Sends a message to a specific agent's inbox."""
        pass

    @abstractmethod
    def read_inbox(self, agent_id: str) -> Dict[str, Any]:
        """Reads a message from a specific agent's inbox."""
        pass

class FilesystemMailboxProvider(MailboxProvider):
    """Filesystem-based mailbox implementation."""

    def __init__(self, session_root: Path):
        self.session_root = session_root
        self.agents_root = session_root / "agents"

    def _get_inbox_path(self, agent_id: str) -> Path:
        return self.agents_root / agent_id / "inbox"

    def send(self, agent_id: str, message: Dict[str, Any]):
        inbox = self._get_inbox_path(agent_id)
        inbox.mkdir(parents=True, exist_ok=True)
        # Unique message name to prevent collision
        # Sanitize ID to prevent directory issues (replace / with _)
        safe_id = str(message.get('id', 'latest')).replace("/", "_")
        path = inbox / f"msg_{safe_id}.json"
        with open(path, "w") as f:
            json.dump(message, f)

    def read_inbox(self, agent_id: str) -> Dict[str, Any]:
        inbox = self._get_inbox_path(agent_id)
        if not inbox.exists():
            return {}
        # Returns the first message found
        for msg_file in inbox.glob("*.json"):
            with open(msg_file, "r") as f:
                return json.load(f)
        return {}

class Mailbox:
    """Convenience wrapper for the MailboxProvider."""
    def __init__(self, provider: MailboxProvider):
        self.provider = provider

    def send(self, agent_id: str, message: Dict[str, Any]):
        self.provider.send(agent_id, message)

    def receive(self, agent_id: str) -> Dict[str, Any]:
        return self.provider.read_inbox(agent_id)
