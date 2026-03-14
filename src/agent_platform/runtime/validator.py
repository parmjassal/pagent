from typing import Dict, Any, Optional, Tuple
import logging
from langchain_openai import ChatOpenAI
from .state import AgentState
from .workspace import WorkspaceContext

logger = logging.getLogger(__name__)

class SystemValidatorAgent:
    """
    Generic System Agent responsible for validating generated code or prompts 
    against session guidelines and guardrail policies.
    """

    def __init__(self, llm: ChatOpenAI, workspace: WorkspaceContext):
        self.llm = llm
        self.workspace = workspace

    def validate_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node to validate the last generated output."""
        
        generated_content = state.get("generated_output")
        if not generated_content:
            return {"messages": [{"role": "system", "content": "Validator: No content to validate."}], "is_valid": True}

        # 1. Extract Guidelines from Session
        session_path = self.workspace.get_session_dir(state["user_id"], state["session_id"])
        # Check for both global level and session level overrides
        guidelines_path = session_path / "guidelines.md"
        
        guidelines = "Use professional tone and ensure safety."
        if guidelines_path.exists():
            guidelines = guidelines_path.read_text()
        else:
            # Fallback to global if session doesn't have it
            global_guidelines = self.workspace.get_global_dir() / "guidelines.md"
            if global_guidelines.exists():
                guidelines = global_guidelines.read_text()

        # 2. Perform Validation (Simulated LLM check)
        # In a real scenario, this would use a structured output chain to return (is_valid, feedback)
        is_valid, reason = self._simulate_validation(generated_content, guidelines)

        log_msg = f"Validator: Result={is_valid}, Reason={reason}"
        logger.info(log_msg)

        return {
            "messages": [{"role": "system", "content": log_msg}],
            "is_valid": is_valid,
            "validation_feedback": reason if not is_valid else None
        }

    def _simulate_validation(self, content: str, guidelines: str) -> Tuple[bool, str]:
        """Placeholder for actual LLM-based validation against Markdown guidelines."""
        # Check if 'destructive' or 'Safety' is mentioned in guidelines
        is_safety_active = "destructive" in guidelines.lower() or "safety" in guidelines.lower()
        
        destructive_keywords = ["delete", "remove", "rm ", "drop", "wipe"]
        if is_safety_active and any(kw in content.lower() for kw in destructive_keywords):
            return False, "Content violates destructive action policy."
        
        return True, "Passed"
