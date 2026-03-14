from pydantic import BaseModel, Field
from typing import List, Optional
from .state import AgentRole

class SubAgentTask(BaseModel):
    agent_id: str = Field(description="Unique ID for the sub-agent")
    role: AgentRole = Field(description="Role of the agent: supervisor or worker")
    instructions: str = Field(description="Specific instructions for this agent")

class DecompositionResult(BaseModel):
    thought_process: str = Field(description="The supervisor's reasoning")
    sub_tasks: List[SubAgentTask] = Field(default_factory=list, description="List of sub-agents to spawn")

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the content adheres to guidelines")
    reasoning: str = Field(description="Detailed explanation of the validation decision")
