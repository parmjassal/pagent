from pathlib import Path
from typing import Optional, Dict, Any
import logging
from ...orch.state import AgentState
from ..context_store import ContextStore
from ..schema import ErrorCode

logger = logging.getLogger(__name__)

class FilesystemTools:
    """
    A collection of core filesystem interaction tools for agents.
    """

    def __init__(self, session_path: Path, context_store: Optional[ContextStore] = None):
        self.session_path = session_path
        self.context_store = context_store

    def _resolve_path(self, path_str: str, state: Optional[AgentState] = None) -> Optional[Path]:
        path = Path(path_str)
        if path.is_absolute():
            resolved = path.resolve()
        else:
            resolved = (self.session_path / path_str).resolve()
        return resolved

    def write_file(self, file_path: str, content: str, state: Optional[AgentState] = None) -> Dict[str, Any]:
        resolved_path = self._resolve_path(file_path)
        if not resolved_path:
            return {"success": False, "error": f"Invalid path: {file_path}", "error_code": ErrorCode.INVALID_ARGUMENTS}
        
        # Restriction: Cannot write to session-level knowledge/ directory directly.
        # Must use ContextTools.update_knowledge for reasoned promotion.
        knowledge_dir = (self.session_path / "knowledge").resolve()
        try:
            if resolved_path == knowledge_dir or knowledge_dir in resolved_path.parents:
                return {
                    "success": False, 
                    "error": "Direct writes to the 'knowledge/' directory are prohibited. Please use 'update_knowledge' to promote global artifacts with proper prefixing.",
                    "error_code": ErrorCode.PERMISSION_DENIED
                }
        except ValueError:
            pass # Paths are on different drives or unrelated

        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(content)
            return {"success": True, "path": str(resolved_path)}
        except Exception as e:
            code = ErrorCode.PERMISSION_DENIED if isinstance(e, PermissionError) else ErrorCode.EXECUTION_ERROR
            return {"success": False, "error": str(e), "error_code": code}

    def read_file(self, file_path: str, state: Optional[AgentState] = None) -> Dict[str, Any]:
        resolved_path = self._resolve_path(file_path)
        if not resolved_path:
            return {"success": False, "error": f"Invalid path: {file_path}", "error_code": ErrorCode.INVALID_ARGUMENTS}
        
        try:
            content = resolved_path.read_text()
            return {"success": True, "content": content}
        except Exception as e:
            code = ErrorCode.PERMISSION_DENIED if isinstance(e, PermissionError) else ErrorCode.EXECUTION_ERROR
            return {"success": False, "error": str(e), "error_code": code}

    def ls(self, path: str = ".", state: Optional[AgentState] = None) -> Dict[str, Any]:
        resolved_path = self._resolve_path(path)
        if not resolved_path or not resolved_path.is_dir():
            return {"success": False, "error": f"Invalid directory: {path}", "error_code": ErrorCode.INVALID_ARGUMENTS}

        try:
            entries = [e.name for e in resolved_path.iterdir()]
            return {"success": True, "entries": entries}
        except Exception as e:
            code = ErrorCode.PERMISSION_DENIED if isinstance(e, PermissionError) else ErrorCode.EXECUTION_ERROR
            return {"success": False, "error": str(e), "error_code": code}
