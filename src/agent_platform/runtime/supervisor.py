from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from .state import AgentState
from .quota import SessionQuota
from .agent_factory import AgentFactory
from .prompt_writer import DynamicPromptWriter
from ..mailbox import Mailbox

class SupervisorAgent:
    """The primary orchestrator that decomposes tasks and spawns sub-agents."""

    def __init__(
        self, 
        agent_factory: AgentFactory,
        mailbox: Mailbox,
        prompt_writer: DynamicPromptWriter,
        model_name: str = "gpt-4o", 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None
    ):
        self.agent_factory = agent_factory
        self.mailbox = mailbox
        self.prompt_writer = prompt_writer
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url
        )

    def _should_spawn_agent(self, state: AgentState) -> str:
        """Determines if more agents need to be spawned or if the task is complete."""
        if state.get("next_steps"):
            return "prompt" # Changed from 'spawn' to 'prompt'
        return END

    def task_decomposition_node(self, state: AgentState) -> AgentState:
        """Decomposes the input task and identifies sub-agents to spawn."""
        return {
            "messages": [{"role": "assistant", "content": "Task decomposition: Need a research agent for deep analysis."}],
            "next_steps": ["researcher_1"] 
        }

    def spawning_node(self, state: AgentState) -> AgentState:
        """Creates a new agent via the factory and sends a message (including the generated prompt) to its inbox."""
        if not state["next_steps"]:
            return state

        sub_agent_id = state["next_steps"][0]
        
        # 1. Check if we can spawn
        new_agent_state = self.agent_factory.create_agent(
            user_id=state["user_id"],
            session_id=state["session_id"],
            agent_id=sub_agent_id,
            current_quota=state["quota"],
            parent_depth=state["current_depth"],
            # Inject the prompt generated in the previous step
            generated_prompt=state.get("generated_prompt")
        )

        if not new_agent_state:
            return {"messages": [{"role": "system", "content": f"Failed to spawn {sub_agent_id}: Quota or Depth exceeded."}]}
        
        # 2. Handover: Write task to sub-agent's inbox
        task_msg = {
            "id": f"task_{state['agent_id']}",
            "sender": state["agent_id"],
            "system_prompt": state.get("generated_prompt"),
            "payload": {"task": "Perform research according to your system prompt."}
        }
        self.mailbox.send(sub_agent_id, task_msg)

        return {
            "quota": SessionQuota(agent_count=1),
            "messages": [{"role": "system", "content": f"Successfully spawned {sub_agent_id} with custom prompt."}],
            "next_steps": [] 
        }

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("decompose", self.task_decomposition_node)
        # Add the Dynamic Prompt Writer node
        workflow.add_node("prompt", self.prompt_writer.generate_prompt_node)
        workflow.add_node("spawn", self.spawning_node)
        
        workflow.set_entry_point("decompose")
        
        workflow.add_conditional_edges(
            "decompose",
            self._should_spawn_agent,
            {
                "prompt": "prompt",
                END: END
            }
        )
        
        # Transition from Prompt Writer to Spawning node
        workflow.add_edge("prompt", "spawn")
        workflow.add_edge("spawn", END)
        
        return workflow.compile()
