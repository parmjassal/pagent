from pydantic import BaseModel, Field
from typing import Annotated
import operator

class SessionQuota(BaseModel):
    """Tracks resource usage within a single session."""
    agent_count: int = Field(default=0, description="Total number of agents spawned in this session")
    message_count: int = Field(default=0, description="Total number of messages exchanged")
    token_usage: int = Field(default=0, description="Total LLM tokens consumed")
    max_agents: int = Field(default=50, description="Maximum allowed agents per session")

    def can_spawn(self) -> bool:
        """Checks if a new agent can be spawned."""
        return self.agent_count < self.max_agents

def update_quota(left: SessionQuota, right: SessionQuota) -> SessionQuota:
    """
    Reducer function for LangGraph to atomically update the SessionQuota.
    It sums up the usage fields while maintaining the constraints.
    """
    return SessionQuota(
        agent_count=left.agent_count + right.agent_count,
        message_count=left.message_count + right.message_count,
        token_usage=left.token_usage + right.token_usage,
        max_agents=left.max_agents # Keeps the limit from the original state
    )

# This type hint will be used in LangGraph State definitions
QuotaState = Annotated[SessionQuota, update_quota]
