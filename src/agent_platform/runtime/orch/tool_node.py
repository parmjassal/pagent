import logging
import uuid
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

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the tool specified in metadata or messages and updates state.
        """
        logger.debug(f"Got the tool call {state}")
        # 1. Resolve tool request from metadata
        tool_call = state.get("metadata", {}).get("next_tool_call")
        if not tool_call:
            return {"messages": [{"role": "tool", "content": "[System] ToolNode: No tool call requested."}]}

        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        
        # Resolve ID from the original LLM call
        tool_call_id = tool_call.get("id")
        
        if not tool_name:
            return {"messages": [{"role": "tool", "content": "[System] ToolNode: Missing tool name."}]}

        # 2. Dispatch Execution (Native or Sandboxed)
        logger.info(f"Agent {state['agent_id']} calling tool: {tool_name} (ID: {tool_call_id or 'none'})")
        result = await self.dispatcher.dispatch(state, tool_name, **tool_args)

        # 3. Process Result
        if result["success"]:
            content = str(result["output"])
            log_msg = f"[System] Tool '{tool_name}' executed successfully via {result.get('source')}."
        else:
            code_suffix = f" (Code: {result.get('error_code')})" if result.get("error_code") else ""
            content = f"Error executing tool '{tool_name}': {result.get('error')}{code_suffix}"
            log_msg = f"[System] Tool '{tool_name}' failed{code_suffix}."

        # 4. Return State Update with Schema Compliance
        # Strict APIs (e.g. ModelArts/OpenAI) require that a 'tool' role message MUST be
        # preceded by an 'assistant' message containing the corresponding 'tool_calls'.
        last_msg = state["messages"][-1] if state["messages"] else {}
        logger.warning(f"Received Message {last_msg}")
        # Normalize last_msg for checking
        last_role = None
        last_tool_calls = []
        if isinstance(last_msg, dict):
            last_role = last_msg.get("role")
            last_tool_calls = last_msg.get("tool_calls", [])
        else:
            last_role = getattr(last_msg, "role", None)
            # AIMessage has tool_calls attribute
            last_tool_calls = getattr(last_msg, "tool_calls", [])

        # Validate if we can safely use the 'tool' role
        can_use_tool_role = False
        if tool_call_id and last_role in ("assistant", "ai") and last_tool_calls:
            # Check if the tool_call_id matches any of the calls in the preceding message
            if any(c.get("id") == tool_call_id if isinstance(c, dict) else getattr(c, "id", None) == tool_call_id for c in last_tool_calls):
                can_use_tool_role = True

        if can_use_tool_role:
            tool_msg = {"role": "tool", "name": tool_name, "content": content, "tool_call_id": tool_call_id}
        else:
            if tool_call_id:
                logger.warning(f"Tool call ID {tool_call_id} provided, but preceding assistant message missing tool_calls metadata. Falling back to 'user' role for result.")
            # Present as a system observation to the agent using 'user' role
            tool_msg = {"role": "tool", "name": tool_name, "content": content, "tool_call_id": tool_call_id}

        return {
            "messages": [
                tool_msg,
                #{"role": "user", "content": log_msg}
            ],
            "metadata": {"next_tool_call": None} # Clear request after execution
        }
