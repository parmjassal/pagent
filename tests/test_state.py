import pytest
from pathlib import Path
from agent_platform.runtime.state import create_initial_state
from agent_platform.runtime.quota import SessionQuota, update_quota

def test_initial_state_creation():
    inbox = Path("/tmp/inbox")
    outbox = Path("/tmp/outbox")
    state = create_initial_state("a1", "u1", "s1", inbox, outbox)
    
    assert state["agent_id"] == "a1"
    assert state["inbox_path"] == inbox
    assert state["quota"].agent_count == 0
    assert state["quota"].max_agents == 50

def test_state_quota_reducer_update():
    # Simulate LangGraph's state update mechanism
    initial_quota = SessionQuota(agent_count=1)
    new_update = SessionQuota(agent_count=2)
    
    # Reducer sums the usage fields
    result = update_quota(initial_quota, new_update)
    assert result.agent_count == 3
