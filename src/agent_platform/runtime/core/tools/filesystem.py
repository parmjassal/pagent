from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FilesystemTools:
    """
    A collection of core filesystem interaction tools for agents.
    All paths should be relative to the session root or explicitly checked
    against the session's allowed workspace boundary.
    """

    def __init__(self, session_path: Path):
        self.session_path = session_path

    def _resolve_path(self, relative_path: str) -> Optional[Path]:
        """Resolves a relative path within the session's allowed boundaries."""
        resolved = (self.session_path / relative_path).resolve()
        # Ensure path is within the session_path (security boundary)
        if not str(resolved).startswith(str(self.session_path)):
            logger.warning(f"Security violation: Attempted to access path outside session boundary: {resolved}")
            return None
        return resolved

    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Writes content to a file within the agent's session workspace.
        Creates parent directories if they don't exist.
        """
        resolved_path = self._resolve_path(file_path)
        if not resolved_path:
            return {"success": False, "error": f"Invalid path: {file_path}. Outside session boundary."}
        
        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(content)
            logger.info(f"File written: {resolved_path}")
            return {"success": True, "path": str(resolved_path)}
        except Exception as e:
            logger.error(f"Error writing file {resolved_path}: {e}")
            return {"success": False, "error": str(e)}

    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Reads content from a file within the agent's session workspace.
        """
        resolved_path = self._resolve_path(file_path)
        if not resolved_path:
            return {"success": False, "error": f"Invalid path: {file_path}. Outside session boundary."}
        
        try:
            content = resolved_path.read_text()
            logger.info(f"File read: {resolved_path}")
            return {"success": True, "content": content}
        except FileNotFoundError:
            return {"success": False, "error": f"File not found: {resolved_path}"}
        except Exception as e:
            logger.error(f"Error reading file {resolved_path}: {e}")
            return {"success": False, "error": str(e)}

    def ls(self, path: str = ".") -> Dict[str, Any]:
        """
        Lists files and directories within a given path relative to the session workspace.
        """
        resolved_path = self._resolve_path(path)
        if not resolved_path:
            return {"success": False, "error": f"Invalid path: {path}. Outside session boundary."}
        if not resolved_path.is_dir():
            return {"success": False, "error": f"Path is not a directory: {resolved_path}"}

        try:
            entries = [e.name for e in resolved_path.iterdir()]
            return {"success": True, "entries": entries}
        except Exception as e:
            logger.error(f"Error listing directory {resolved_path}: {e}")
            return {"success": False, "error": str(e)}
