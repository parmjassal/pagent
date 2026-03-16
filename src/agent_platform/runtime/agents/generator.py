from enum import Enum
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..orch.state import AgentState
from ..core.workspace import WorkspaceContext
from ..core.http_client import get_platform_http_client

import logging
logger = logging.getLogger(__name__)

class TaskType(str, Enum):
    PROMPT = "prompt"
    TOOL = "tool"

class SystemGeneratorAgent:
    """
    Generic System Agent responsible for generating context-aware 
    Code or Prompts using external templates.
    """

    def __init__(
        self, 
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        llm: Optional[Any] = None, 
        workspace: Optional[WorkspaceContext] = None
    ):
        self.workspace = workspace
        
        if llm:
            self.llm = llm
        else:
            http_client = get_platform_http_client()
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                http_client=http_client,
                temperature=0.7 # Higher temperature for creative generation
            )

    async def generate_node(self, state: AgentState, task_type: TaskType = TaskType.PROMPT) -> Dict[str, Any]:
        """LangGraph node to generate tailored output using session templates."""
        
        target_id = "unknown"
        if state.get("next_steps"):
            target_id = state["next_steps"][0]
        elif state.get("messages"):
            last_msg = state["messages"][-1]["content"]
            if "spawned " in last_msg:
                target_id = last_msg.split("spawned ")[1].strip()

        # 1. Resolve Template from Session
        if self.workspace:
            session_path = self.workspace.get_session_dir(state["user_id"], state["session_id"])
        else:
            # Fallback for tests if workspace not injected
            session_path = state["inbox_path"].parent.parent.parent 
            
        template_name = f"generator_{task_type.value}.txt"
        template_path = session_path / "prompts" / template_name
        
        system_instruction = f"Generate a {task_type.value}."
        if template_path.exists():
            system_instruction = template_path.read_text()

        # Resolve Task Context from metadata
        task_context = state.get("metadata", {}).get("current_task_instructions", "No specific instructions provided.")

        # 2. Invoke LLM with structured messages
        # System Message contains the "How to Generate"
        # Human Message contains the "What to Generate For"
        human_data = f"Target Agent ID: {target_id}\nTarget Task Description: {task_context}"
        
        logger.debug(f"Generator Input ({task_type.value}): {human_data}")
        
        response = await self.llm.ainvoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=human_data)
        ])
        content = response.content if hasattr(response, "content") else str(response)

        logger.debug(f"Generator Output ({task_type.value}): {content}")
        log_msg = f"Generator: Created {task_type.value} for {target_id}"
        return {
            "messages": [{"role": "system", "content": log_msg}],
            "generated_output": content
        }
