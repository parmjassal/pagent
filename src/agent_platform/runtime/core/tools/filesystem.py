from pathlib import Path
from typing import Optional, Dict, Any
import logging
from ...orch.state import AgentState

logger = logging.getLogger(__name__)

class FilesystemTools:
    """
    A collection of core filesystem interaction tools for agents.
    """

    def __init__(self, session_path: Path):
        self.session_path = session_path

    def _resolve_path(self, relative_path: str) -> Optional[Path]:
        resolved = (self.session_path / relative_path).resolve()
        if not str(resolved).startswith(str(self.session_path)):
            logger.warning(f"Security violation: Attempted to access path outside session boundary: {resolved}")
            return None
        return resolved

    def write_file(self, file_path: str, content: str, state: Optional[AgentState] = None) -> Dict[str, Any]:
        resolved_path = self._resolve_path(file_path)
        if not resolved_path:
            return {"success": False, "error": f"Invalid path: {file_path}"}
        
        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(content)
            return {"success": True, "path": str(resolved_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_file(self, file_path: str, state: Optional[AgentState] = None) -> Dict[str, Any]:
        resolved_path = self._resolve_path(file_path)
        if not resolved_path:
            return {"success": False, "error": f"Invalid path: {file_path}"}
        
        try:
            content = resolved_path.read_text()
            return {"success": True, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def ls(self, path: str = ".", state: Optional[AgentState] = None) -> Dict[str, Any]:
        resolved_path = self._resolve_path(path)
        if not resolved_path or not resolved_path.is_dir():
            return {"success": False, "error": f"Invalid directory: {path}"}

        try:
            entries = [e.name for e in resolved_path.iterdir()]
            return {"success": True, "entries": entries}
        except Exception as e:
            return {"success": False, "error": str(e)}
