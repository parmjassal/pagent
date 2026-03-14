from typing import Dict, Any, Optional, Tuple, Union
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from .state import AgentState
from .workspace import WorkspaceContext
from .models import ValidationResult

logger = logging.getLogger(__name__)

class SystemValidatorAgent:
    """
    Generic System Agent responsible for validating generated code or prompts 
    against session guidelines.
    """

    def __init__(self, llm: Optional[Any] = None, workspace: Optional[WorkspaceContext] = None):
        # Injected LLM should be configured with .with_structured_output(ValidationResult)
        self.llm = llm
        self.workspace = workspace

    def validate_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node to validate the last generated output."""
        
        generated_content = state.get("generated_output")
        if not generated_content:
            return {"messages": [{"role": "system", "content": "Validator: No content."}], "is_valid": True}

        # 1. Resolve Guidelines
        guidelines = "Use professional tone and ensure safety."
        if self.workspace:
            session_path = self.workspace.get_session_dir(state["user_id"], state["session_id"])
            guidelines_path = session_path / "guidelines.md"
            if guidelines_path.exists():
                guidelines = guidelines_path.read_text()

        # 2. Invoke LLM (Mocked or Real)
        instruction = f"Validate this content against these guidelines:\n\nContent: {generated_content}\n\nGuidelines: {guidelines}"
        
        # This returns a ValidationResult object (or a mock)
        result: ValidationResult = self.llm.invoke([SystemMessage(content=instruction)])

        log_msg = f"Validator: is_valid={result.is_valid}, reason='{result.reasoning}'"
        logger.info(log_msg)

        return {
            "messages": [{"role": "system", "content": log_msg}],
            "is_valid": result.is_valid,
            "validation_feedback": result.reasoning if not result.is_valid else None
        }
