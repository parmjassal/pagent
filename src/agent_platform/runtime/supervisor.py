from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from .state import AgentState
from .quota import SessionQuota
from .agent_factory import AgentFactory
from ..mailbox import Mailbox

class SupervisorAgent:
    """The primary orchestrator that decomposes tasks and spawns sub-agents."""

    def __init__(
        self, 
        agent_factory: AgentFactory,
        mailbox: Mailbox,
        model_name: str = "gpt-4o", 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None
    ):
        self.agent_factory = agent_factory
        self.mailbox = mailbox
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
        """Creates a new agent via the factory and sends a message to its inbox."""
        if not state["next_steps"]:
            return state

        sub_agent_id = state["next_steps"][0]
        
        # 1. Check if we can spawn (Factory handles both quota and depth)
        new_agent_state = self.agent_factory.create_agent(
            user_id=state["user_id"],
            session_id=state["session_id"],
            agent_id=sub_agent_id,
            current_quota=state["quota"],
            parent_depth=state["current_depth"]
        )

        if not new_agent_state:
            return {"messages": [{"role": "system", "content": f"Failed to spawn {sub_agent_id}: Quota or Depth exceeded."}]}
        
        # 2. Handover: Write task to sub-agent's inbox
        task_msg = {
            "id": f"task_{state['agent_id']}",
            "sender": state["agent_id"],
            "payload": {"task": "Decomposed task content"} # In reality, from decomposition
        }
        self.mailbox.send(sub_agent_id, task_msg)

        # 3. Update state (Increment agent count via reducer)
        return {
            "quota": SessionQuota(agent_count=1),
            "messages": [{"role": "system", "content": f"Successfully spawned and messaged: {sub_agent_id}"}],
            "next_steps": [] 
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
