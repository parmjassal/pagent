import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class KnowledgeManager(ABC):
    """
    Interface for persistent knowledge extraction and retrieval.
    Abstracts how 'Fact Sheets' are stored (Filesystem, Vector DB, etc).
    """
    @abstractmethod
    def store_fact(self, key: str, content: str, source_pointer: str):
        """Stores an extracted fact or summary."""
        pass

    @abstractmethod
    def retrieve_fact(self, key: str) -> Optional[str]:
        """Retrieves a specific fact by its key/ID."""
        pass

    @abstractmethod
    def list_facts(self) -> List[str]:
        """Returns IDs of all known facts in the current context."""
        pass

class FilesystemKnowledgeManager(KnowledgeManager):
    """
    Implementation of 'Option B': Markdown-based Fact Sheets stored in the filesystem.
    """
    def __init__(self, knowledge_root: Path):
        self.root = knowledge_root
        self.root.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace(" ", "_")
        return self.root / f"{safe_key}.md"

    def store_fact(self, key: str, content: str, source_pointer: str):
        path = self._get_path(key)
        markdown_content = f"# Fact: {key}\nSource: `{source_pointer}`\n\n{content}\n"
        path.write_text(markdown_content)
        logger.info(f"Knowledge stored: {path}")

    def retrieve_fact(self, key: str) -> Optional[str]:
        path = self._get_path(key)
        if path.exists():
            return path.read_text()
        return None

    def list_facts(self) -> List[str]:
        return [f.stem for f in self.root.glob("*.md")]
