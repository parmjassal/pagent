from enum import Enum
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .state import AgentState
from .workspace import WorkspaceContext

class TaskType(str, Enum):
    PROMPT = "prompt"
    TOOL = "tool"

class SystemGeneratorAgent:
    """
    Generic System Agent responsible for generating context-aware 
    Code or Prompts based on the specified TaskType.
    """

    def __init__(self, llm: ChatOpenAI, workspace: WorkspaceContext):
        self.llm = llm
        self.workspace = workspace

    def generate_node(self, state: AgentState, task_type: TaskType = TaskType.PROMPT) -> Dict[str, Any]:
        """LangGraph node to generate tailored output (Prompt or Python Tool)."""
        
        # 1. Resolve target identification
        target_id = "unknown"
        if state.get("next_steps"):
            target_id = state["next_steps"][0]
        elif state.get("messages"):
            # Try to infer target from the last 'assistant' message (e.g. 'Successfully spawned researcher_1')
            last_msg = state["messages"][-1]["content"]
            if "spawned " in last_msg:
                target_id = last_msg.split("spawned ")[1].strip()

        session_dir = self.workspace.get_session_dir(state["user_id"], state["session_id"])
        
        # 1. Resolve Instructions based on TaskType
        instruction_template = "You are a specialized AI assistant."
        if task_type == TaskType.TOOL:
            instruction_template = "Write a Python function that performs the requested task. Return ONLY the code."
        
        # 2. Use LLM to generate
        # In a real scenario, we'd pull templates from session_dir / "prompts" / f"{task_type}_template.txt"
        
        # Simulated generation
        if task_type == TaskType.TOOL:
            generated = f"def {target_id}_func(x): return f'Result for {{x}}'"
            log_msg = f"Generator: Created Python tool for {target_id}"
        else:
            generated = f"SYSTEM PROMPT for {target_id}: Accomplish the task."
            log_msg = f"Generator: Created system prompt for {target_id}"

        return {
            "messages": [{"role": "system", "content": log_msg}],
            "generated_output": generated # Generic field for the output
        }
