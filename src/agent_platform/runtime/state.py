from typing import TypedDict, Annotated, List, Dict, Any, Optional, Union
import operator
from pathlib import Path
from .quota import QuotaState, SessionQuota

def update_next_steps(left: List[str], right: Union[List[str], None]) -> List[str]:
    if right is None:
        return left
    if not right:
        return []
    return left + right

def update_counts(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    """Reducer to accumulate node visit counts."""
    new_counts = left.copy()
    for k, v in right.items():
        new_counts[k] = new_counts.get(k, 0) + v
    return new_counts

class AgentState(TypedDict):
    """
    The base state for all agents in the platform.
    """
    agent_id: str
    user_id: str
    session_id: str
    inbox_path: Path
    outbox_path: Path
    quota: QuotaState
    current_depth: int
    generated_output: Optional[str]
    messages: Annotated[List[Dict[str, Any]], operator.add]
    next_steps: Annotated[List[str], update_next_steps]
    # Loop Detection Fields
    node_counts: Annotated[Dict[str, int], update_counts]
    metadata: Annotated[Dict[str, Any], operator.ior] # Merges dicts

def create_initial_state(
    agent_id: str, 
    user_id: str, 
    session_id: str, 
    inbox_path: Path, 
    outbox_path: Path,
    current_depth: int = 0,
    max_agents: int = 50,
    generated_output: Optional[str] = None
) -> AgentState:
    """Helper to initialize a new agent state."""
    return {
        "agent_id": agent_id,
        "user_id": user_id,
        "session_id": session_id,
        "inbox_path": inbox_path,
        "outbox_path": outbox_path,
        "quota": SessionQuota(max_agents=max_agents),
        "current_depth": current_depth,
        "generated_output": generated_output,
        "messages": [],
        "next_steps": [],
        "node_counts": {},
        "metadata": {}
    }
