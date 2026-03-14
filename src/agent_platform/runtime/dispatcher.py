from enum import Enum
from typing import Dict, Any, Callable, Optional
import logging
from .sandbox import SandboxRunner, SandboxResult
from .guardrails import GuardrailManager
from .state import AgentState

logger = logging.getLogger(__name__)

class ToolSource(str, Enum):
    COMMUNITY = "community" # Native execution
    DYNAMIC = "dynamic"     # Sandboxed execution

def _dynamic_execution_stub(**kwargs):
    """Module-level helper to ensure serializability for the Sandbox process."""
    return f"Sandboxed result with {kwargs}"

class ToolDispatcher:
    """
    Dispatches tool calls to either Native or Sandboxed execution 
    based on the tool's registered source.
    """

    def __init__(
        self, 
        sandbox: SandboxRunner, 
        guardrails: GuardrailManager
    ):
        self.sandbox = sandbox
        self.guardrails = guardrails
        self.registry: Dict[str, ToolSource] = {}
        self.native_tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, source: ToolSource, func: Optional[Callable] = None):
        """Registers a tool name and its source (and implementation if native)."""
        self.registry[name] = source
        if func and source == ToolSource.COMMUNITY:
            self.native_tools[name] = func

    def dispatch(self, state: AgentState, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Orchestrates: Guardrail Check -> Dispatch -> Execution.
        """
        is_allowed, reason = self.guardrails.validate_tool_call(state, tool_name, kwargs)
        if not is_allowed:
            return {"error": f"Guardrail blocked tool: {reason}", "success": False}

        source = self.registry.get(tool_name, ToolSource.DYNAMIC)

        if source == ToolSource.COMMUNITY:
            return self._execute_native(state, tool_name, **kwargs)
        else:
            return self._execute_sandboxed(state, tool_name, **kwargs)

    def _execute_native(self, state: AgentState, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Runs community tools in the main process."""
        logger.info(f"Executing NATIVE tool: {tool_name}")
        func = self.native_tools.get(tool_name)
        if not func:
            return {"error": f"Native tool {tool_name} not found in registry", "success": False}
        
        try:
            result = func(**kwargs)
            return {"output": result, "success": True, "source": "native"}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _execute_sandboxed(self, state: AgentState, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Runs dynamic tools in the process sandbox."""
        logger.info(f"Executing SANDBOXED tool: {tool_name}")
        
        # In this skeleton, we use the serializable stub function
        result: SandboxResult = self.sandbox.run(_dynamic_execution_stub, **kwargs)
        
        if result.success:
            return {"output": result.output, "success": True, "source": "sandbox", "duration": result.duration}
        else:
            return {"error": result.error, "success": False, "source": "sandbox"}
