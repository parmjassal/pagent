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
        result_hook: ResultHook,
        model_config: Optional[Dict[str, Any]] = None
    ):
        self.factory = agent_factory
        self.mailbox = mailbox
        self.generator = generator
        self.dispatcher = dispatcher
        self.result_hook = result_hook
        self.model_config = model_config or {"model_name": "gpt-4o"}

    def compile_unit(self, role: AgentRole, checkpointer: Optional[BaseCheckpointSaver] = None) -> Any:
        """Builds and compiles the graph for the specific role."""
        
        if role == AgentRole.SUPERVISOR:
            agent = SupervisorAgent(
                self.factory, self.mailbox, self.generator,
                unit_compiler=self, # Self-reference for recursion
                **self.model_config
            )
            tool_node = AgentToolNode(self.dispatcher)
            return agent.build_graph(checkpointer=checkpointer, tool_node=tool_node)
        
        elif role == AgentRole.WORKER:
            tool_node = AgentToolNode(self.dispatcher)
            agent = WorkerAgent(
                tool_node,
                **self.model_config
            )
            return agent.build_graph()
        
        else:
            raise ValueError(f"Unknown AgentRole: {role}")

import asyncio
