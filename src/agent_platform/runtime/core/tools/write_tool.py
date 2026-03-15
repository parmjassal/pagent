from typing import Dict, Any
from pathlib import Path
from ..agents.generator import SystemGeneratorAgent, TaskType
from ..agents.validator import SystemValidatorAgent
from ..dispatcher import ToolRegistry
from ..state import AgentState

class WriteTool:
    """
    A meta-tool that enables agents to generate, validate, and register new tools.
    """
    def __init__(
        self, 
        generator: SystemGeneratorAgent, 
        validator: SystemValidatorAgent, 
        registry: ToolRegistry
    ):
        self.generator = generator
        self.validator = validator
        self.registry = registry

    async def __call__(
        self,
        state: AgentState,
        tool_name: str, 
        task_description: str
    ) -> Dict[str, Any]:
        """
        Generates, validates, and registers a new tool in the session.
        """
        # 1. Generate Tool Code
        state["next_steps"] = [tool_name]
        state["metadata"]["current_task_instructions"] = task_description
        
        gen_result = await self.generator.generate_node(state, task_type=TaskType.TOOL)
        tool_code = gen_result["generated_output"]
        
        # 2. Validate Generated Code
        state["generated_output"] = tool_code
        val_result = await self.validator.validate_node(state)
        
        if not val_result["is_valid"]:
            return {"error": f"Validation failed: {val_result['validation_feedback']}", "success": False}
        
        # 3. Persist and Register
        skills_path = self.registry.session_path / "skills"
        skills_path.mkdir(exist_ok=True)
        code_path = skills_path / f"{tool_name}.py"
        code_path.write_text(tool_code)
        
        self.registry.register_dynamic(tool_name, task_description, code_path)
        
        return {"status": f"Tool '{tool_name}' created and registered successfully.", "success": True}
