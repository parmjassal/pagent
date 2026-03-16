from typing import Dict, Any, Optional, List
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from ..orch.state import AgentState, AgentRole
from ..orch.tool_node import AgentToolNode
from ..orch.models import WorkerResult, ExecutionStrategy
from ..orch.logic import LoopMonitor
from ..core.http_client import get_platform_http_client
from ..core.parser import robust_json_parser

logger = logging.getLogger(__name__)

class WorkerAgent:
    """
    Specialized Worker agent that reason about tasks and uses tools to execute.
    """

    def __init__(
        self, 
        tool_node: AgentToolNode,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        llm: Optional[Any] = None,
        tool_manifest: Optional[str] = None,
        result_hook: Optional[Any] = None
    ):
        self.tool_node = tool_node
        self.tool_manifest = tool_manifest
        self.result_hook = result_hook
        self.parser = JsonOutputParser(pydantic_object=WorkerResult)
        
        if llm:
            self.llm = llm
        else:
            http_client = get_platform_http_client()
            self.base_llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                http_client=http_client,
                temperature=0 
            )
            self.llm = self.base_llm | self.parser

    def _should_continue(self, state: AgentState) -> str:
        """Determines the next path based on the strategy."""
        strategy = state.get("metadata", {}).get("strategy")
        
        # Loop Check
        if LoopMonitor.check_node_loop(state, "reason", threshold=100):
            return "abort"

        if strategy == ExecutionStrategy.TOOL_USE:
            return "tools"
        elif strategy == ExecutionStrategy.FINISH:
            return END
        
        # If strategy is missing (due to parsing error in reasoning_node), 
        # retry reasoning (up to threshold).
        return "reason"

    async def reasoning_node(self, state: AgentState) -> AgentState:
        """Invokes the LLM to decide on a tool-use or finish strategy."""
        
        system_instruction = "You are a specialized worker agent. Use tools to complete your task."
        
        # Inject Tool Manifest if available
        if self.tool_manifest:
            system_instruction = f"{system_instruction}\n\n{self.tool_manifest}"

        format_instructions = self.parser.get_format_instructions()
        full_instruction = f"{system_instruction}\n\n{format_instructions}"

        prompt = [
            SystemMessage(content=full_instruction),
            *state["messages"]
        ]
        
        logger.debug(f"Prompt is {prompt}")
        response = await self.llm.ainvoke(prompt)
        
        # Robust Parsing
        if hasattr(response, "content"):
            # If it's an AIMessage, parse the content string
            try:
                parsed = robust_json_parser(response.content)
                result = WorkerResult.model_validate(parsed)
            except Exception as e:
                logger.error(f"Worker failed to parse JSON: {response.content}")
                # Provide a corrective message back to the LLM
                return {
                    "messages": [
                        {"role": "assistant", "content": response.content},
                        {"role": "user", "content": f"[System] Error: Your response could not be parsed as JSON. You MUST return ONLY a valid JSON object conforming to the schema. Do not include any text outside the JSON block. Error detail: {e}"}
                    ],
                    "node_counts": {"reason": 1}
                }
        else:
            # Already a dict or object
            result = WorkerResult.model_validate(response)
        
        metadata_update = {
            "strategy": result.strategy,
            "thought_process": result.thought_process
        }

        if result.strategy == ExecutionStrategy.TOOL_USE and result.tool_call:
            # Capture ID from AIMessage if it was a real tool call format
            tool_call_id = None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_call_id = response.tool_calls[0].get("id")
            
            tc_dump = result.tool_call.model_dump()
            if tool_call_id:
                tc_dump["id"] = tool_call_id
                
            metadata_update["next_tool_call"] = tc_dump
            
            assistant_msg = {"role": "assistant", "content": result.thought_process}
            if tool_call_id:
                # Reconstruct tool_calls metadata for history schema compliance
                assistant_msg["tool_calls"] = [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": result.tool_call.name,
                        "arguments": json.dumps(result.tool_call.args)
                    }
                }]

            return {
                "role": state["role"],
                "messages": [assistant_msg],
                "metadata": metadata_update,
                "node_counts": {"reason": 1}
            }

        final_answer = result.final_answer or "Task finished."
        if self.result_hook:
            # ResultHook expects (agent_id, result)
            processed_result = self.result_hook.process_result(state["agent_id"], final_answer)
        else:
            processed_result = {"type": "inline", "content": final_answer}

        return {
            "role": state["role"],
            "messages": [{"role": "assistant", "content": result.thought_process}],
            "final_result": processed_result,
            "metadata": metadata_update,
            "node_counts": {"reason": 1}
        }

    def abort_node(self, state: AgentState) -> Dict[str, Any]:
        return {"messages": [{"role": "user", "content": "[System] Worker ABORTING: Loop detected."}]}

    def build_graph(self, checkpointer: Optional[Any] = None) -> Any:
        workflow = StateGraph(AgentState)
        workflow.add_node("reason", self.reasoning_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("abort", self.abort_node)
        
        workflow.set_entry_point("reason")
        
        workflow.add_conditional_edges(
            "reason",
            self._should_continue,
            {
                "tools": "tools",
                "abort": "abort",
                END: END
            }
        )
        
        workflow.add_edge("tools", "reason") # Loop back after tool execution
        workflow.add_edge("abort", END)
        
        return workflow.compile(checkpointer=checkpointer)
