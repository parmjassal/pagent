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
    generated_output: Optional[str] # General field for Generator output
    messages: Annotated[List[Dict[str, Any]], operator.add]
    next_steps: Annotated[List[str], update_next_steps]

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
    """Helper to initialize a new agent state with default quota values."""
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
        "next_steps": []
    }
