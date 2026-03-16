from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
from enum import Enum
from .state import AgentRole

class ExecutionStrategy(str, Enum):
    DECOMPOSE = "decompose"
    TOOL_USE = "tool_use"
    FINISH = "finish"

class SubAgentTask(BaseModel):
    agent_id: str = Field(description="Unique ID for the sub-agent")
    role: AgentRole = Field(description="Role of the agent: supervisor or worker")
    instructions: str = Field(description="Specific instructions for this agent")

class ToolCall(BaseModel):
    id: Optional[str] = Field(default=None, description="The unique ID of the tool call from the LLM")
    name: str = Field(description="Name of the tool to invoke")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")

class PlanningResult(BaseModel):
    thought_process: str = Field(description="The supervisor's reasoning for the chosen strategy")
    strategy: ExecutionStrategy = Field(description="The chosen path: decompose, tool_use, or finish")
    sub_tasks: Optional[List[SubAgentTask]] = Field(default=None, description="Required if strategy is 'decompose'")
    tool_call: Optional[ToolCall] = Field(default=None, description="Required if strategy is 'tool_use'")

class WorkerResult(BaseModel):
    thought_process: str = Field(description="The worker's reasoning")
    strategy: ExecutionStrategy = Field(description="Either 'tool_use' or 'finish'")
    tool_call: Optional[ToolCall] = Field(default=None, description="Required if strategy is 'tool_use'")
    final_answer: Optional[str] = Field(default=None, description="The final answer if strategy is 'finish'")

class DecompositionResult(BaseModel):
    thought_process: str = Field(description="The supervisor's reasoning")
    sub_tasks: List[SubAgentTask] = Field(default_factory=list, description="List of sub-agents to spawn")

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the content adheres to guidelines")
    reasoning: str = Field(description="Detailed explanation of the validation decision")
