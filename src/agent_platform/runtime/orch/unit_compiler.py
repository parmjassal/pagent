from typing import Any, Dict, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from .state import AgentState, AgentRole
from ..agents.orchestrator import OrchestratorAgent
from ..orch.tool_node import AgentToolNode
from .result_hook import ResultHook

class UnitCompiler:
    """
    Dynamically compiles LangGraph 'Units' (Orchestrator) 
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
        """Builds and compiles the graph for the specific role using OrchestratorAgent."""
        
        # Initialize LLM for the unit
        from langchain_openai import ChatOpenAI
        from ..core.http_client import get_platform_http_client
        
        http_client = get_platform_http_client()
        llm = ChatOpenAI(
            model=self.model_config.get("model_name", "gpt-4o"),
            openai_api_base=self.model_config.get("openai_base_url"),
            http_client=http_client,
            temperature=0
        )

        # In Unified Model v3.0, both Supervisor and Worker use OrchestratorAgent.
        # The prompt template (loaded in planner_node) determines the behavior.
        agent = OrchestratorAgent(
            self.factory, self.mailbox, self.generator,
            llm=llm,
            unit_compiler=self, # Self-reference for recursion
            result_hook=self.result_hook,
            tool_manifest=self.dispatcher.registry.get_tool_manifest()
        )
        tool_node = AgentToolNode(self.dispatcher)
        return agent.build_graph(checkpointer=checkpointer, tool_node=tool_node)

import asyncio
