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
    Code or Prompts.
    """

    def __init__(self, llm: Optional[Any] = None, workspace: Optional[WorkspaceContext] = None):
        # Allow injection for testing
        self.llm = llm
        self.workspace = workspace

    def generate_node(self, state: AgentState, task_type: TaskType = TaskType.PROMPT) -> Dict[str, Any]:
        """LangGraph node to generate tailored output (Prompt or Python Tool)."""
        
        # 1. Resolve target identification
        target_id = "unknown"
        if state.get("next_steps"):
            target_id = state["next_steps"][0]
        elif state.get("messages"):
            last_msg = state["messages"][-1]["content"]
            if "spawned " in last_msg:
                target_id = last_msg.split("spawned ")[1].strip()

        # 2. Invoke LLM (Mocked or Real)
        # In production, we'd pass detailed instructions and session context
        instruction = f"Generate a {task_type.value} for target: {target_id}"
        
        # This will now trigger the injected mock or a real LLM call
        response = self.llm.invoke([SystemMessage(content=instruction)])
        
        # Response might be a BaseMessage (real) or a string (simple mock)
        content = response.content if hasattr(response, "content") else str(response)

        log_msg = f"Generator: Created {task_type.value} for {target_id}"

        return {
            "messages": [{"role": "system", "content": log_msg}],
            "generated_output": content
        }
