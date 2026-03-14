from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from .state import AgentState
from .quota import SessionQuota
from .agent_factory import AgentFactory
from .generator import SystemGeneratorAgent, TaskType
from .logic import LoopMonitor
from ..mailbox import Mailbox

class SupervisorAgent:
    """The primary orchestrator that decomposes tasks and spawns sub-agents."""

    def __init__(
        self, 
        agent_factory: AgentFactory,
        mailbox: Mailbox,
        generator: SystemGeneratorAgent,
        model_name: str = "gpt-4o", 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None
    ):
        self.agent_factory = agent_factory
        self.mailbox = mailbox
        self.generator = generator
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url
        )

    def _should_continue(self, state: AgentState) -> str:
        """Centralized router with Loop Detection."""
        # 1. Check for infinite loops
        if LoopMonitor.check_node_loop(state, "decompose") or LoopMonitor.check_content_loop(state):
            return "abort"
            
        # 2. Check for next steps
        if state.get("next_steps"):
            return "generate_prompt"
            
        return END

    def task_decomposition_node(self, state: AgentState) -> AgentState:
        update = {
            "messages": [{"role": "assistant", "content": "Task decomposition: Need a research agent."}],
            "next_steps": ["researcher_1"] 
        }
        # Track node visit
        update.update(LoopMonitor.get_update("decompose"))
        return update

    def generate_prompt_node(self, state: AgentState) -> Dict[str, Any]:
        return self.generator.generate_node(state, task_type=TaskType.PROMPT)

    def spawning_node(self, state: AgentState) -> AgentState:
        if not state["next_steps"]:
            return state

        sub_agent_id = state["next_steps"][0]
        prompt = state.get("generated_output")

        new_agent_state = self.agent_factory.create_agent(
            user_id=state["user_id"],
            session_id=state["session_id"],
            agent_id=sub_agent_id,
            current_quota=state["quota"],
            parent_depth=state["current_depth"],
            generated_output=prompt
        )

        if not new_agent_state:
            return {"messages": [{"role": "system", "content": f"Failed to spawn {sub_agent_id}"}]}
        
        task_msg = {
            "id": f"task_{state['agent_id']}",
            "sender": state["agent_id"],
            "system_prompt": prompt,
            "payload": {"task": "Perform research."}
        }
        self.mailbox.send(sub_agent_id, task_msg)

        return {
            "quota": SessionQuota(agent_count=1),
            "messages": [{"role": "system", "content": f"Successfully spawned {sub_agent_id}"}],
            "next_steps": [] 
        }

    def abort_node(self, state: AgentState) -> Dict[str, Any]:
        """Safety node to stop the graph when a loop is detected."""
        return {"messages": [{"role": "system", "content": "ABORTING: Infinite loop detected in execution."}]}

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("decompose", self.task_decomposition_node)
        workflow.add_node("generate_prompt", self.generate_prompt_node)
        workflow.add_node("spawn", self.spawning_node)
        workflow.add_node("abort", self.abort_node)
        
        workflow.set_entry_point("decompose")
        
        # Routing after decomposition
        workflow.add_conditional_edges(
            "decompose", 
            self._should_continue, 
            {
                "generate_prompt": "generate_prompt", 
                "abort": "abort",
                END: END
            }
        )
        
        workflow.add_edge("generate_prompt", "spawn")
        workflow.add_edge("spawn", END)
        workflow.add_edge("abort", END)
        
        return workflow.compile()
