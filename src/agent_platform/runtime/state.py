from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
from pathlib import Path
from .quota import QuotaState, SessionQuota

class AgentState(TypedDict):
    """
    The base state for all agents in the platform.
    
    Attributes:
        agent_id: The unique identifier for the agent instance.
        user_id: The ID of the user owning the session.
        session_id: The ID of the active session.
        inbox_path: Absolute path to the agent's mailbox inbox.
        outbox_path: Absolute path to the agent's mailbox outbox.
        quota: Shared session-level resource tracking (atomic updates).
        messages: The conversation history for this specific agent.
        next_steps: A list of tasks or sub-agents to spawn (used by Supervisor).
    """
    agent_id: str
    user_id: str
    session_id: str
    inbox_path: Path
    outbox_path: Path
    quota: QuotaState
    current_depth: int
    generated_prompt: Optional[str]
    messages: Annotated[List[Dict[str, Any]], operator.add]
    next_steps: Annotated[List[str], operator.add]

def create_initial_state(
    agent_id: str, 
    user_id: str, 
    session_id: str, 
    inbox_path: Path, 
    outbox_path: Path,
    current_depth: int = 0,
    max_agents: int = 50,
    generated_prompt: Optional[str] = None
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
        "generated_prompt": generated_prompt,
        "messages": [],
        "next_steps": []
    }
