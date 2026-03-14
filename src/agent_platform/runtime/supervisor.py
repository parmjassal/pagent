from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from .state import AgentState, AgentRole
from .quota import SessionQuota
from .agent_factory import AgentFactory
from .generator import SystemGeneratorAgent, TaskType
from .logic import LoopMonitor
from .http_client import get_platform_http_client
from .models import DecompositionResult
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
        base_url: Optional[str] = None,
        llm: Optional[Any] = None # New: Allow injection for testing
    ):
        self.agent_factory = agent_factory
        self.mailbox = mailbox
        self.generator = generator
        
        if llm:
            self.llm = llm
        else:
            http_client = get_platform_http_client()
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                http_client=http_client
            ).with_structured_output(DecompositionResult)

    def _should_continue(self, state: AgentState) -> str:
        node_threshold = 3 if state["role"] == AgentRole.SUPERVISOR else 10
        if LoopMonitor.check_node_loop(state, "decompose", threshold=node_threshold):
            return "abort"
        if LoopMonitor.check_content_loop(state, window=3):
            return "abort"
        if state.get("next_steps"):
            return "generate_prompt"
        return END

    def task_decomposition_node(self, state: AgentState) -> AgentState:
        """Invokes the LLM to decompose the task into sub-agents."""
        # Note: In a real system, we'd include system guidelines here
        prompt = [
            SystemMessage(content="You are a task decomposition supervisor. Analyze the user request and identify sub-agents to spawn."),
            *state["messages"]
        ]
        
        # This will now use the structured output model
        result: DecompositionResult = self.llm.invoke(prompt)
        
        next_steps = [task.agent_id for task in result.sub_tasks]
        
        # Prepare metadata for the next agent role (taking the first one)
        next_role = result.sub_tasks[0].role if result.sub_tasks else AgentRole.WORKER
        instructions = result.sub_tasks[0].instructions if result.sub_tasks else ""

        return {
            "messages": [{"role": "assistant", "content": result.thought_process}],
            "next_steps": next_steps,
            "metadata": {
                "next_agent_role": next_role,
                "current_task_instructions": instructions
            },
            "node_counts": {"decompose": 1}
        }

    def generate_prompt_node(self, state: AgentState) -> Dict[str, Any]:
        return self.generator.generate_node(state, task_type=TaskType.PROMPT)

    def spawning_node(self, state: AgentState) -> AgentState:
        if not state["next_steps"]:
            return state

        sub_agent_id = state["next_steps"][0]
        prompt = state.get("generated_output")
        next_role = state.get("metadata", {}).get("next_agent_role", AgentRole.WORKER)

        new_agent_state = self.agent_factory.create_agent(
            user_id=state["user_id"],
            session_id=state["session_id"],
            agent_id=sub_agent_id,
            current_quota=state["quota"],
            parent_depth=state["current_depth"],
            generated_output=prompt,
            role=next_role 
        )

        if not new_agent_state:
            return {"messages": [{"role": "system", "content": f"Failed to spawn {sub_agent_id}"}]}
        
        self.mailbox.send(sub_agent_id, {
            "id": f"task_{state['agent_id']}",
            "sender": state["agent_id"],
            "system_prompt": prompt,
            "payload": {"instructions": state.get("metadata", {}).get("current_task_instructions")},
            "role": next_role
        })

        return {
            "quota": SessionQuota(agent_count=1),
            "messages": [{"role": "system", "content": f"Successfully spawned {next_role.value}: {sub_agent_id}"}],
            "next_steps": [] 
        }

    def abort_node(self, state: AgentState) -> Dict[str, Any]:
        return {"messages": [{"role": "system", "content": f"ABORTING: Loop detected for {state['role']} agent."}]}

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("decompose", self.task_decomposition_node)
        workflow.add_node("generate_prompt", self.generate_prompt_node)
        workflow.add_node("spawn", self.spawning_node)
        workflow.add_node("abort", self.abort_node)
        workflow.set_entry_point("decompose")
        workflow.add_conditional_edges("decompose", self._should_continue, {"generate_prompt": "generate_prompt", "abort": "abort", END: END})
        workflow.add_edge("generate_prompt", "spawn")
        workflow.add_edge("spawn", END)
        workflow.add_edge("abort", END)
        return workflow.compile()
