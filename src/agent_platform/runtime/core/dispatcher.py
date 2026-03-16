import json
import logging
import inspect
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional, List, Union
from .sandbox import SandboxRunner, SandboxResult
from .guardrails import GuardrailManager
from ..orch.state import AgentState
from .schema import ToolSource, ErrorCode, ErrorDetail

logger = logging.getLogger(__name__)

class DynamicToolLoader(ABC):
    @abstractmethod
    def get_executable(self, name: str, code_path: Optional[str] = None) -> Callable:
        pass

class ToolRegistry:
    """
    Manages and persists the tool manifest for a session.
    Uses Python introspection (inspect) to automatically discover tool signatures.
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
        try:
            self.registry_file.write_text(json.dumps(self.metadata, indent=2))
        except Exception as e:
            logger.error(f"Failed to save tool registry: {e}")

    def register_native(self, name: str, func: Callable, summary: str = "", source: ToolSource = ToolSource.COMMUNITY):
        """
        Registers a native Python function as a tool.
        Automatically discovers parameters using 'inspect'.
        """
        # 1. Automatic Parameter Discovery
        try:
            sig = inspect.signature(func)
            params = []
            for param_name in sig.parameters:
                # Skip internal platform arguments
                if param_name not in ("self", "state"):
                    params.append(param_name)
        except ValueError:
            # Fallback for built-ins or non-introspectable objects
            params = []

        # 2. Automatic Summary Discovery from Docstring
        if not summary and func.__doc__:
            summary = func.__doc__.strip().split('\n')[0]

        # 3. Update Metadata
        self.metadata[name] = {
            "source": source,
            "summary": summary or "No description.",
            "parameters": params
        }
        self.native_funcs[name] = func
        self._save()

    def register_langchain_tool(self, lc_tool: Any):
        """
        Wraps a LangChain BaseTool and registers it.
        Maps LC's 'args' and 'description' to the platform manifest.
        """
        name = lc_tool.name
        summary = lc_tool.description
        
        # Discover parameters from LangChain's args schema
        params = []
        if hasattr(lc_tool, "args"):
            params = list(lc_tool.args.keys())

        self.metadata[name] = {
            "source": ToolSource.COMMUNITY,
            "summary": summary,
            "parameters": params
        }

        # Wrap LC tool in a compatible native call that injects kwargs
        async def wrapped_call(state: AgentState, **kwargs):
            # LangChain tools use invoke/ainvoke
            if hasattr(lc_tool, "ainvoke"):
                return await lc_tool.ainvoke(kwargs)
            else:
                return lc_tool.invoke(kwargs)
        
        self.native_funcs[name] = wrapped_call
        self._save()

    def register_dynamic(self, name: str, summary: str, code_path: Path):
        """Registers a tool that will be executed in a sandbox."""
        self.metadata[name] = {
            "source": ToolSource.DYNAMIC,
            "summary": summary,
            "path": str(code_path),
            "parameters": ["..."] # Dynamic tools specify their own args in code
        }
        self._save()

    def get_source(self, name: str) -> ToolSource:
        entry = self.metadata.get(name, {})
        source_val = entry.get("source", ToolSource.DYNAMIC)
        return ToolSource(source_val)

    def get_tool_manifest(self) -> str:
        """Returns a Markdown-formatted list of available tools for prompt injection."""
        nl = chr(10)
        manifest = f"## Available Tools{nl}{nl}"
        
        # Sort by source (CORE first) then name for consistency
        sorted_tools = sorted(
            self.metadata.items(), 
            key=lambda x: (0 if x[1].get("source") == ToolSource.CORE else 1, x[0])
        )
        
        for name, meta in sorted_tools:
            params_list = meta.get("parameters", [])
            params_str = f"({', '.join(params_list)})" if params_list else "()"
            manifest += f"- **{name}{params_str}**: {meta.get('summary', 'No description.')}{nl}"
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

    async def dispatch(self, state: AgentState, tool_name: str, **kwargs) -> Dict[str, Any]:
        # Guardrail check
        is_allowed, reason = await self.guardrails.validate_tool_call(state, tool_name, kwargs)
        if not is_allowed:
            return ErrorDetail(
                code=ErrorCode.GUARDRAIL_BLOCK,
                message=f"Guardrail blocked: {reason}",
                details={"reason": reason}
            ).to_dict()

        source = self.registry.get_source(tool_name)
        if source == ToolSource.COMMUNITY or source == ToolSource.CORE:
            return await self._execute_native(tool_name, state, **kwargs)
        else:
            return self._execute_sandboxed(tool_name, **kwargs)

    async def _execute_native(self, tool_name: str, state: AgentState, **kwargs) -> Dict[str, Any]:
        func = self.registry.native_funcs.get(tool_name)
        if not func:
            return ErrorDetail(
                code=ErrorCode.TOOL_NOT_FOUND,
                message=f"Native tool {tool_name} not found"
            ).to_dict()
        try:
            # Handle both async and sync native tools
            if inspect.iscoroutinefunction(func):
                result = await func(state=state, **kwargs)
            else:
                # Check if it accepts 'state' (our native tools do, wrapped LC tools might not directly)
                # But our wrapped_call above handles passing kwargs to LC.
                # For direct native registration, we expect (state, **kwargs)
                result = func(state=state, **kwargs)
            
            if isinstance(result, dict) and not result.get("success", True):
                if "error_code" not in result:
                    result["error_code"] = ErrorCode.EXECUTION_ERROR.value
                return result

            return {"output": result, "success": True, "source": "native"}
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            return ErrorDetail(
                code=ErrorCode.EXECUTION_ERROR,
                message=str(e),
                details={"exception": type(e).__name__}
            ).to_dict()

    def _execute_sandboxed(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if not self.dynamic_loader:
            return ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="Dynamic tool loader not configured"
            ).to_dict()

        entry = self.registry.metadata.get(tool_name, {})
        executable = self.dynamic_loader.get_executable(tool_name, entry.get("path"))
        
        result: SandboxResult = self.sandbox.run(executable, **kwargs)
        
        if result.success:
            return {"output": result.output, "success": True, "source": "sandbox"}
            
        code = ErrorCode.EXECUTION_ERROR
        if result.error and "Timeout" in result.error:
            code = ErrorCode.TIMEOUT
            
        return ErrorDetail(
            code=code,
            message=result.error or "Unknown sandbox error"
        ).to_dict()
