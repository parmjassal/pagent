from typing import Any, Dict, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from .state import AgentState, AgentRole
from ..agents.supervisor import SupervisorAgent
from ..agents.worker import WorkerAgent
from ..orch.tool_node import AgentToolNode
from .result_hook import ResultHook

class UnitCompiler:
    """
    Dynamically compiles LangGraph 'Units' (Worker or Supervisor) 
    based on the assigned AgentRole.
    """

    def __init__(
        self, 
        agent_factory: Any, 
        mailbox: Any, 
        generator: Any,
        dispatcher: Any,
        result_hook: ResultHook
    ):
        self.factory = agent_factory
        self.mailbox = mailbox
        self.generator = generator
        self.dispatcher = dispatcher
        self.result_hook = result_hook

    def _wrap_final_result(self, node_func: Callable) -> Callable:
        """Helper to wrap a node so its output passes through the ResultHook."""
        async def wrapped(state: AgentState) -> Dict[str, Any]:
            update = await node_func(state) if asyncio.iscoroutinefunction(node_func) else node_func(state)
            
            # If the node is producing a result that marks completion
            if update.get("final_result"):
                update["final_result"] = self.result_hook.process_result(
                    state["agent_id"], update["final_result"]
                )
            return update
        return wrapped

    def compile_unit(self, role: AgentRole, checkpointer: Optional[BaseCheckpointSaver] = None) -> Any:
        """Builds and compiles the graph for the specific role."""
        
        if role == AgentRole.SUPERVISOR:
            agent = SupervisorAgent(self.factory, self.mailbox, self.generator)
            return agent.build_graph(checkpointer=checkpointer)
        
        elif role == AgentRole.WORKER:
            tool_node = AgentToolNode(self.dispatcher)
            agent = WorkerAgent(tool_node)
            return agent.build_graph()
        
        else:
            raise ValueError(f"Unknown AgentRole: {role}")

import asyncio
