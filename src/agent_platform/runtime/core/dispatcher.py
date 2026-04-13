import json
import logging
import inspect
import copy
import weakref
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional, List, Union

# Assuming these imports exist in your project structure
from .sandbox import SandboxRunner, SandboxResult
from .guardrails import GuardrailManager
from ..orch.state import AgentState
from .schema import ToolSource, ErrorCode, ErrorDetail
from ..storage.todo_tool import TODOTool

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
        
        # KEY CHANGE: Tie tool instance lifecycle to the AgentState object.
        # When AgentState is GC'd, the tool copy is GC'd.
        #self._agent_specific_tools = weakref.WeakKeyDictionary()

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
        """Registers a native Python function as a tool."""
        try:
            sig = inspect.signature(func)
            params = [p for p in sig.parameters if p not in ("self", "state")]
        except ValueError:
            params = []

        if not summary and func.__doc__:
            summary = func.__doc__.strip().split('\n')[0]

        self.metadata[name] = {
            "source": source,
            "summary": summary or "No description.",
            "parameters": params
        }
        self.native_funcs[name] = func
        self._save()

    def register_langchain_tool(self, lc_tool: Any):
        """
        Wraps a LangChain BaseTool. 
        Uses WeakKeyDictionary to ensure the tool instance lives exactly as long as the AgentState.
        """
        name = lc_tool.name
        summary = lc_tool.description
        params = list(lc_tool.args.keys()) if hasattr(lc_tool, "args") else []

        self.metadata[name] = {
            "source": ToolSource.COMMUNITY,
            "summary": summary,
            "parameters": params
        }

        async def wrapped_call(state: AgentState, **kwargs):
            if name == "python_repl_ast":
                # 1. Retrieve or create the agent-specific tool fork
                # The 'state' object is the key, ensuring isolation for parallel agents.
                
                # Create a fork of the baseline tool
                agent_tool_instance = copy.copy(lc_tool)
                
                # Resolve path from this specific agent's state
                todo_path = Path(state.get("todo_path")).parent
                todo_tool = TODOTool(todo_path)
                
                # Setup isolated globals for this agent
                agent_globals = dict(lc_tool.globals or {})
                agent_globals["add_task"] = todo_tool.add_task
                
                agent_tool_instance.globals = agent_globals
                agent_tool_instance.locals = agent_globals

                if hasattr(agent_tool_instance, "ainvoke"):
                    return await agent_tool_instance.ainvoke(kwargs)
                return agent_tool_instance.invoke(kwargs)

            # Default for non-REPL tools
            if hasattr(lc_tool, "ainvoke"):
                return await lc_tool.ainvoke(kwargs)
            return lc_tool.invoke(kwargs)
        
        self.native_funcs[name] = wrapped_call
        self._save()

    def register_dynamic(self, name: str, summary: str, code_path: Path):
        self.metadata[name] = {
            "source": ToolSource.DYNAMIC,
            "summary": summary,
            "path": str(code_path),
            "parameters": ["..."]
        }
        self._save()

    def get_source(self, name: str) -> ToolSource:
        entry = self.metadata.get(name, {})
        return ToolSource(entry.get("source", ToolSource.DYNAMIC))

    def get_tool_manifest(self) -> str:
        nl = chr(10)
        manifest = f"## Available Tools{nl}{nl}"
        sorted_tools = sorted(
            self.metadata.items(), 
            key=lambda x: (0 if x[1].get("source") == ToolSource.CORE else 1, x[0])
        )
        for name, meta in sorted_tools:
            params = meta.get("parameters", [])
            manifest += f"- **{name}({', '.join(params)})**: {meta.get('summary', 'No description.')}{nl}"
        return manifest

class ToolDispatcher:
    def __init__(self, registry: ToolRegistry, sandbox: SandboxRunner, guardrails: GuardrailManager, dynamic_loader: Optional[DynamicToolLoader] = None):
        self.registry = registry
        self.sandbox = sandbox
        self.guardrails = guardrails
        self.dynamic_loader = dynamic_loader

    async def dispatch(self, state: AgentState, tool_name: str, **kwargs) -> Dict[str, Any]:
        is_allowed, reason = await self.guardrails.validate_tool_call(state, tool_name, kwargs)
        if not is_allowed:
            return ErrorDetail(code=ErrorCode.GUARDRAIL_BLOCK, message=f"Blocked: {reason}").to_dict()

        source = self.registry.get_source(tool_name)
        if source in (ToolSource.COMMUNITY, ToolSource.CORE):
            return await self._execute_native(tool_name, state, **kwargs)
        return self._execute_sandboxed(tool_name, **kwargs)

    async def _execute_native(self, tool_name: str, state: AgentState, **kwargs) -> Dict[str, Any]:
        func = self.registry.native_funcs.get(tool_name)
        if not func:
            return ErrorDetail(code=ErrorCode.TOOL_NOT_FOUND, message=f"Tool {tool_name} not found").to_dict()
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(state=state, **kwargs)
            else:
                result = func(state=state, **kwargs)
            return {"output": result, "success": True, "source": "native"}
        except Exception as e:
            logger.exception(f"Error in {tool_name}")
            return ErrorDetail(code=ErrorCode.EXECUTION_ERROR, message=str(e)).to_dict()

    def _execute_sandboxed(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if not self.dynamic_loader:
            return ErrorDetail(code=ErrorCode.INTERNAL_ERROR, message="No loader").to_dict()
        entry = self.registry.metadata.get(tool_name, {})
        executable = self.dynamic_loader.get_executable(tool_name, entry.get("path"))
        result = self.sandbox.run(executable, **kwargs)
        if result.success:
            return {"output": result.output, "success": True, "source": "sandbox"}
        return ErrorDetail(code=ErrorCode.EXECUTION_ERROR, message=result.error or "Error").to_dict()