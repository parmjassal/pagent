import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List
from .sandbox import SandboxRunner, SandboxResult
from .guardrails import GuardrailManager
from .state import AgentState

logger = logging.getLogger(__name__)

class ToolSource(str, Enum):
    COMMUNITY = "community" # Native execution (LangChain)
    DYNAMIC = "dynamic"     # Sandboxed execution (System Generator)

def _dynamic_execution_stub(**kwargs):
    """Module-level helper for Sandbox serialization."""
    return f"Sandboxed result with {kwargs}"

class ToolRegistry:
    """
    Manages the lifecycle and metadata of tools within a session.
    Persists dynamic tool metadata to ensure they are remembered across restarts.
    """

    def __init__(self, session_path: Path):
        self.session_path = session_path
        self.registry_file = session_path / "tool_registry.json"
        self.metadata: Dict[str, Dict[str, Any]] = self._load()
        self.native_funcs: Dict[str, Callable] = {}

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.registry_file.exists():
            try:
                return json.loads(self.registry_file.read_text())
            except Exception as e:
                logger.error(f"Failed to load tool registry: {e}")
        return {}

    def _save(self):
        self.registry_file.write_text(json.dumps(self.metadata, indent=2))

    def register_native(self, name: str, func: Callable):
        """Registers a community tool for native execution."""
        self.metadata[name] = {"source": ToolSource.COMMUNITY}
        self.native_funcs[name] = func
        # We don't necessarily persist native tools as they are re-registered on bootstrap

    def register_dynamic(self, name: str, code_path: Optional[Path] = None):
        """Registers a system-generated tool for sandboxed execution."""
        self.metadata[name] = {
            "source": ToolSource.DYNAMIC,
            "path": str(code_path) if code_path else None
        }
        self._save()

    def get_source(self, name: str) -> ToolSource:
        entry = self.metadata.get(name, {})
        return ToolSource(entry.get("source", ToolSource.DYNAMIC))

class ToolDispatcher:
    """
    Orchestrates tool execution using the ToolRegistry and SandboxRunner.
    """

    def __init__(
        self, 
        registry: ToolRegistry,
        sandbox: SandboxRunner, 
        guardrails: GuardrailManager
    ):
        self.registry = registry
        self.sandbox = sandbox
        self.guardrails = guardrails

    def dispatch(self, state: AgentState, tool_name: str, **kwargs) -> Dict[str, Any]:
        # 1. Guardrail Validation
        is_allowed, reason = self.guardrails.validate_tool_call(state, tool_name, kwargs)
        if not is_allowed:
            return {"error": f"Guardrail blocked: {reason}", "success": False}

        # 2. Identify Source from Registry
        source = self.registry.get_source(tool_name)

        # 3. Execute
        if source == ToolSource.COMMUNITY:
            return self._execute_native(tool_name, **kwargs)
        else:
            return self._execute_sandboxed(tool_name, **kwargs)

    def _execute_native(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        func = self.registry.native_funcs.get(tool_name)
        if not func:
            return {"error": f"Native tool {tool_name} implementation not found", "success": False}
        
        try:
            result = func(**kwargs)
            return {"output": result, "success": True, "source": "native"}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _execute_sandboxed(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        logger.info(f"Dispatching to SANDBOX: {tool_name}")
        # Placeholder for dynamic code loading from self.registry.metadata[tool_name]['path']
        result: SandboxResult = self.sandbox.run(_dynamic_execution_stub, **kwargs)
        
        if result.success:
            return {"output": result.output, "success": True, "source": "sandbox"}
        return {"error": result.error, "success": False}
