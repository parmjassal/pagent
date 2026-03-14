import pytest
from pathlib import Path
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.lifecycle import AgentLifecycleManager
from agent_platform.runtime.quota import SessionQuota

@pytest.fixture
def setup_context(tmp_path):
    workspace = WorkspaceContext(root=tmp_path)
    factory = AgentFactory(workspace)
    lifecycle = AgentLifecycleManager(workspace, factory)
    return workspace, lifecycle

def test_lifecycle_archive_agent(setup_context):
    workspace, lifecycle = setup_context
    user_id, session_id, agent_id = "u1", "s1", "a1"
    quota = SessionQuota()

    # Create agent
    lifecycle.create_agent(user_id, session_id, agent_id, quota)
    agent_dir = workspace.get_agent_dir(user_id, session_id, agent_id)
    (agent_dir / "inbox" / "msg.json").write_text("{}")

    assert agent_dir.exists()

    # Archive agent
    lifecycle.archive_agent(user_id, session_id, agent_id)
    
    assert not agent_dir.exists()
    
    session_dir = workspace.get_session_dir(user_id, session_id)
    archive_dir = session_dir / "archive" / "agents" / agent_id
    assert archive_dir.exists()
    assert (archive_dir / "inbox" / "msg.json").exists()

def test_list_active_agents(setup_context):
    workspace, lifecycle = setup_context
    user_id, session_id = "u1", "s1"
    quota = SessionQuota()

    lifecycle.create_agent(user_id, session_id, "agent1", quota)
    lifecycle.create_agent(user_id, session_id, "agent2", quota)

    active_agents = lifecycle.list_active_agents(user_id, session_id)
    assert len(active_agents) == 2
    assert "agent1" in active_agents
    assert "agent2" in active_agents
