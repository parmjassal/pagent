from typing import Dict, Any, Optional, Tuple, Union
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from ..orch.state import AgentState
from ..core.workspace import WorkspaceContext
from ..orch.models import ValidationResult

logger = logging.getLogger(__name__)

class SystemValidatorAgent:
    """
    Generic System Agent responsible for validating generated code or prompts 
    using external session templates and resilient parsing.
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
        
        # ALWAYS initialize the parser
        self.parser = JsonOutputParser(pydantic_object=ValidationResult)
        
        if llm:
            self.llm = llm
        else:
            self.base_llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0
            )
            self.llm = self.base_llm | self.parser

    async def validate_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node to validate the last generated output."""
        
        generated_content = state.get("generated_output")
        if not generated_content:
            return {"messages": [{"role": "system", "content": "Validator: No content."}], "is_valid": True}

        # 1. Resolve Template from Session
        session_path = state["inbox_path"].parent.parent.parent
        template_path = session_path / "prompts" / "validator_check.txt"
        
        system_instruction = "You are a Security and Quality Auditor."
        if template_path.exists():
            system_instruction = template_path.read_text()

        # 2. Resolve Session Guidelines
        guidelines = "Use professional tone and ensure safety."
        guidelines_path = session_path / "guidelines.md"
        if guidelines_path.exists():
            guidelines = guidelines_path.read_text()

        # 3. Invoke LLM with Resilient Parsing
        format_instructions = self.parser.get_format_instructions()
        instruction = f"{system_instruction}\n\n{format_instructions}\n\nValidate this content against these guidelines:\n\nContent: {generated_content}\n\nGuidelines: {guidelines}"
        
        logger.debug(f"Validator Input: {instruction}")
        
        raw_result = await self.llm.ainvoke([SystemMessage(content=instruction)])
        result = ValidationResult.model_validate(raw_result)

        logger.debug(f"Validator Result: {result.model_dump_json()}")
        log_msg = f"Validator: is_valid={result.is_valid}, reason='{result.reasoning}'"
        logger.info(log_msg)

        return {
            "messages": [{"role": "system", "content": log_msg}],
            "is_valid": result.is_valid,
            "validation_feedback": result.reasoning if not result.is_valid else None
        }
