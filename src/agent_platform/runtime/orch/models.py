from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
from enum import Enum
from .state import AgentRole

class ExecutionStrategy(str, Enum):
    AUTHORIZE = "authorize"
    DECOMPOSE = "decompose"
    TOOL_USE = "tool_use"
    FINISH = "finish"

class SubAgentTask(BaseModel):
    agent_id: str = Field(description="Unique ID for the sub-agent")
    role: AgentRole = Field(description="Role of the agent: supervisor or worker")
    instructions: str = Field(description="Specific instructions for this agent")

class Action(BaseModel):
    strategy: ExecutionStrategy = Field(description="The chosen path: authorize, decompose, tool_use, or finish")
    name: Optional[str] = Field(default=None, description="Tool name (required for authorize/tool_use)")
    args: Optional[Dict[str, Any]] = Field(default=None, description="Tool arguments")
    sub_tasks: Optional[List[SubAgentTask]] = Field(default=None, description="List of sub-tasks (required for decompose)")
    final_answer: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="The final result (required for finish)")

class PlanningResult(BaseModel):
    thought_process: str = Field(description="The reasoning for the chosen strategy/action sequence")
    action_sequence: List[Action] = Field(default_factory=list, description="Sequence of actions to execute")

class WorkerResult(BaseModel):
    thought_process: str = Field(description="The worker's reasoning")
    action_sequence: List[Action] = Field(default_factory=list, description="Sequence of actions to execute")

class DecompositionResult(BaseModel):
    thought_process: str = Field(description="The supervisor's reasoning")
    sub_tasks: List[SubAgentTask] = Field(default_factory=list, description="List of sub-agents to spawn")

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the content adheres to guidelines")
    reasoning: str = Field(description="Detailed explanation of the validation decision")
