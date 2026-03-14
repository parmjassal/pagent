import pytest
from pathlib import Path
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.quota import SessionQuota

@pytest.fixture
def workspace(tmp_path):
    return WorkspaceContext(root=tmp_path)

def test_create_agent_success(workspace):
    factory = AgentFactory(workspace)
    quota = SessionQuota(agent_count=0, max_agents=10)
    
    user_id = "user1"
    session_id = "sess1"
    agent_id = "agent1"
    
    state = factory.create_agent(user_id, session_id, agent_id, quota)
    
    assert state is not None
    assert state["agent_id"] == "agent1"
    assert state["quota"].agent_count == 1 # Added by factory for new agent
    assert (workspace.get_agent_dir(user_id, session_id, agent_id) / "inbox").exists()

def test_create_agent_quota_exceeded(workspace):
    factory = AgentFactory(workspace)
    # Quota already at max
    quota = SessionQuota(agent_count=5, max_agents=5)
    
    state = factory.create_agent("u1", "s1", "a1", quota)
    assert state is None
