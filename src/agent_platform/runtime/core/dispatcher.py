import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional, List
from .sandbox import SandboxRunner, SandboxResult
from .guardrails import GuardrailManager
from ..orch.state import AgentState
from .tool_registry_defaults import STATIC_TOOL_REGISTRY
from .schema import ToolSource

logger = logging.getLogger(__name__)

class DynamicToolLoader(ABC):
    @abstractmethod
    def get_executable(self, name: str, code_path: Optional[str] = None) -> Callable:
        pass

class ToolRegistry:
    """
    Manages and persists the tool manifest for a session.
    """
    def __init__(self, session_path: Path):
        self.session_path = session_path
        self.registry_file = session_path / "tool_registry.json"
        self.metadata: Dict[str, Dict[str, Any]] = self._load()
        self.native_funcs: Dict[str, Callable] = {}
        
        # Load static defaults on init
        self._load_defaults()

    def _load_defaults(self):
        for tool in STATIC_TOOL_REGISTRY:
            if tool["name"] not in self.metadata:
                self.metadata[tool["name"]] = {"source": tool["source"], "summary": tool["summary"]}

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.registry_file.exists():
            return json.loads(self.registry_file.read_text())
        return {}

    def _save(self):
        self.registry_file.write_text(json.dumps(self.metadata, indent=2))

    def register_native(self, name: str, func: Callable, summary: str = ""):
        self.metadata[name] = {"source": ToolSource.COMMUNITY, "summary": summary}
        self.native_funcs[name] = func

    def register_dynamic(self, name: str, summary: str, code_path: Path):
        self.metadata[name] = {
            "source": ToolSource.DYNAMIC,
            "summary": summary,
            "path": str(code_path)
        }
        self._save()

    def get_source(self, name: str) -> ToolSource:
        entry = self.metadata.get(name, {})
        return ToolSource(entry.get("source", ToolSource.DYNAMIC))

    def get_tool_manifest(self) -> str:
        """Returns a Markdown-formatted list of available tools for prompt injection."""
        nl = chr(10)
        manifest = f"## Available Tools{nl}{nl}"
        for name, meta in self.metadata.items():
            manifest += f"- **{name}**: {meta.get('summary', 'No description.')}{nl}"
        return manifest

class ToolDispatcher:
    """
    Orchestrates tool execution using the ToolRegistry and SandboxRunner.
    """

    def __init__(
        self, 
        registry: ToolRegistry,
        sandbox: SandboxRunner, 
        guardrails: GuardrailManager,
        dynamic_loader: Optional[DynamicToolLoader] = None
    ):
        self.registry = registry
        self.sandbox = sandbox
        self.guardrails = guardrails
        self.dynamic_loader = dynamic_loader

    def dispatch(self, state: AgentState, tool_name: str, **kwargs) -> Dict[str, Any]:
        is_allowed, reason = self.guardrails.validate_tool_call(state, tool_name, kwargs)
        if not is_allowed:
            return {"error": f"Guardrail blocked: {reason}", "success": False}

        source = self.registry.get_source(tool_name)

        if source == ToolSource.COMMUNITY or source == ToolSource.CORE:
            return self._execute_native(tool_name, **kwargs)
        else:
            return self._execute_sandboxed(tool_name, **kwargs)

    def _execute_native(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        func = self.registry.native_funcs.get(tool_name)
        if not func:
            return {"error": f"Native tool {tool_name} not found", "success": False}
        try:
            result = func(**kwargs)
            return {"output": result, "success": True, "source": "native"}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _execute_sandboxed(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        logger.info(f"Dispatching to SANDBOX: {tool_name}")
        
        if not self.dynamic_loader:
            return {"error": "Dynamic tool loader not configured", "success": False}

        entry = self.registry.metadata.get(tool_name, {})
        executable = self.dynamic_loader.get_executable(tool_name, entry.get("path"))
        
        result: SandboxResult = self.sandbox.run(executable, **kwargs)
        
        if result.success:
            return {"output": result.output, "success": True, "source": "sandbox"}
        return {"error": result.error, "success": False}
