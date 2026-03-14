from enum import Enum
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..orch.state import AgentState
from ..core.workspace import WorkspaceContext

class TaskType(str, Enum):
    PROMPT = "prompt"
    TOOL = "tool"

class SystemGeneratorAgent:
    """
    Generic System Agent responsible for generating context-aware 
    Code or Prompts.
    """

    def __init__(self, llm: Optional[Any] = None, workspace: Optional[WorkspaceContext] = None):
        self.llm = llm
        self.workspace = workspace

    def generate_node(self, state: AgentState, task_type: TaskType = TaskType.PROMPT) -> Dict[str, Any]:
        """LangGraph node to generate tailored output (Prompt or Python Tool)."""
        target_id = "unknown"
        if state.get("next_steps"):
            target_id = state["next_steps"][0]
        elif state.get("messages"):
            last_msg = state["messages"][-1]["content"]
            if "spawned " in last_msg:
                target_id = last_msg.split("spawned ")[1].strip()

        instruction = f"Generate a {task_type.value} for target: {target_id}"
        response = self.llm.invoke([SystemMessage(content=instruction)])
        content = response.content if hasattr(response, "content") else str(response)

        log_msg = f"Generator: Created {task_type.value} for {target_id}"
        return {
            "messages": [{"role": "system", "content": log_msg}],
            "generated_output": content
        }
