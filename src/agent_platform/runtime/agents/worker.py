from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from ..orch.state import AgentState
from ..orch.tool_node import AgentToolNode
from .generator import SystemGeneratorAgent

class WorkerAgent:
    """
    Standard Worker agent that executes specialized tasks using tools.
    """

    def __init__(self, tool_node: AgentToolNode):
        self.tool_node = tool_node

    def _should_call_tool(self, state: AgentState) -> str:
        """Logic to decide whether to call a tool or finish."""
        if state.get("metadata", {}).get("next_tool_call"):
            return "tools"
        return END

    def build_graph(self) -> Any:
        workflow = StateGraph(AgentState)
        
        # In a real worker, we might have a 'reasoning' node first
        # But here we show the tool-use integration
        workflow.add_node("tools", self.tool_node)
        
        workflow.set_entry_point("tools") # For this skeleton, it starts with tool call
        
        workflow.add_conditional_edges(
            "tools",
            self._should_call_tool,
            {
                "tools": "tools",
                END: END
            }
        )
        
        return workflow.compile()
