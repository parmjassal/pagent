from enum import Enum
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Union
import operator
from pathlib import Path
from .quota import QuotaState, SessionQuota

class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    WORKER = "worker"
    SYSTEM = "system"

def update_next_steps(left: List[str], right: Union[List[str], None]) -> List[str]:
    if right is None:
        return left
    if not right:
        return []
    return left + right

def update_counts(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    new_counts = left.copy()
    for k, v in right.items():
        new_counts[k] = new_counts.get(k, 0) + v
    return new_counts

def update_interactions(left: List[str], right: Union[List[str], None]) -> List[str]:
    """Custom reducer for active_interactions. [] clears the list."""
    if right is None:
        return left
    if not right:
        return []
    return left + right

class AgentState(TypedDict):
    """
    The base state for all agents in the platform.
    """
    agent_id: str
    role: AgentRole
    user_id: str
    session_id: str
    inbox_path: Path
    outbox_path: Path
    knowledge_path: Path
    todo_path: Path
    quota: QuotaState
    current_depth: int
    generated_output: Optional[str]
    final_result: Optional[Dict[str, Any]] # New: Processed Result
    messages: Annotated[List[Dict[str, Any]], operator.add]
    next_steps: Annotated[List[str], update_next_steps]
    node_counts: Annotated[Dict[str, int], update_counts]
    active_interactions: Annotated[List[str], update_interactions]
    metadata: Annotated[Dict[str, Any], operator.ior]

def create_initial_state(
    agent_id: str, 
    user_id: str, 
    session_id: str, 
    inbox_path: Path, 
    outbox_path: Path,
    knowledge_path: Optional[Path] = None,
    todo_path: Optional[Path] = None,
    role: AgentRole = AgentRole.WORKER,
    current_depth: int = 0,
    max_agents: int = 50,
    generated_output: Optional[str] = None
) -> AgentState:
    """Helper to initialize a new agent state."""
    return {
        "agent_id": agent_id,
        "role": role,
        "user_id": user_id,
        "session_id": session_id,
        "inbox_path": inbox_path,
        "outbox_path": outbox_path,
        "knowledge_path": knowledge_path or Path("/tmp/knowledge"),
        "todo_path": todo_path or Path("/tmp/todo"),
        "quota": SessionQuota(max_agents=max_agents),
        "current_depth": current_depth,
        "generated_output": generated_output,
        "final_result": None,
        "messages": [],
        "next_steps": [],
        "node_counts": {},
        "active_interactions": [],
        "metadata": {}
    }
