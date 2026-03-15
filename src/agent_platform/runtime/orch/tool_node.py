import logging
from typing import Dict, Any, List, Optional
from ..orch.state import AgentState
from ..core.dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

class AgentToolNode:
    """
    Standardized LangGraph node for executing tools via the ToolDispatcher.
    Handles results, errors, and state updates for any agent.
    """

    def __init__(self, dispatcher: ToolDispatcher):
        self.dispatcher = dispatcher

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the tool specified in metadata or messages and updates state.
        """
        # 1. Resolve tool request from metadata
        tool_call = state.get("metadata", {}).get("next_tool_call")
        if not tool_call:
            return {"messages": [{"role": "system", "content": "ToolNode: No tool call requested."}]}

        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})

        if not tool_name:
            return {"messages": [{"role": "system", "content": "ToolNode: Missing tool name."}]}

        # 2. Dispatch Execution (Native or Sandboxed)
        logger.info(f"Agent {state['agent_id']} calling tool: {tool_name}")
        result = self.dispatcher.dispatch(state, tool_name, **tool_args)

        # 3. Process Result
        if result["success"]:
            content = str(result["output"])
            log_msg = f"Tool '{tool_name}' executed successfully via {result.get('source')}."
        else:
            content = f"Error executing tool '{tool_name}': {result.get('error')}"
            log_msg = f"Tool '{tool_name}' failed."

        # 4. Return State Update
        return {
            "messages": [
                {"role": "tool", "name": tool_name, "content": content},
                {"role": "system", "content": log_msg}
            ],
            "metadata": {"next_tool_call": None} # Clear request after execution
        }
