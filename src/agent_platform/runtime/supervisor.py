from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from .state import AgentState
from .quota import SessionQuota

class SupervisorAgent:
    """The primary orchestrator that decomposes tasks and spawns sub-agents."""

    def __init__(
        self, 
        model_name: str = "gpt-4o", 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url
        )

    def _should_spawn_agent(self, state: AgentState) -> str:
        """Determines if more agents need to be spawned or if the task is complete."""
        if state.get("next_steps"):
            return "spawn"
        return END

    def task_decomposition_node(self, state: AgentState) -> AgentState:
        """Decomposes the input task and identifies sub-agents to spawn."""
        # Simple placeholder for decomposition logic
        # In a real scenario, this would call self.llm to analyze the task
        return {
            "messages": [{"role": "system", "content": "Decomposing task..."}],
            "next_steps": ["research_agent"] # Example sub-agent to spawn
        }

    def spawning_node(self, state: AgentState) -> AgentState:
        """Checks quota and increments agent count (Spawning simulation)."""
        if not state["quota"].can_spawn():
            return {"messages": [{"role": "system", "content": "Quota exceeded, cannot spawn sub-agent."}]}
        
        # Incrementing quota via reducer (agent_count=1 will be added to existing)
        return {
            "quota": SessionQuota(agent_count=1),
            "messages": [{"role": "system", "content": f"Spawning agent: {state['next_steps'][0]}"}],
            "next_steps": [] # Clear next steps after processing
        }

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("decompose", self.task_decomposition_node)
        workflow.add_node("spawn", self.spawning_node)
        
        workflow.set_entry_point("decompose")
        
        workflow.add_conditional_edges(
            "decompose",
            self._should_spawn_agent,
            {
                "spawn": "spawn",
                END: END
            }
        )
        
        workflow.add_edge("spawn", END)
        
        return workflow.compile()
