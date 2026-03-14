import pytest
from pathlib import Path
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.quota import SessionQuota
from agent_platform.runtime.supervisor import SupervisorAgent
from agent_platform.runtime.state import create_initial_state
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def workspace(tmp_path):
    return WorkspaceContext(root=tmp_path)

@pytest.fixture
def mailbox(tmp_path):
    provider = FilesystemMailboxProvider(tmp_path / "sess1")
    return Mailbox(provider)

def test_supervisor_spawning_logic(workspace, mailbox):
    # Setup Factory with depth limit 1
    factory = AgentFactory(workspace, max_spawn_depth=1)
    supervisor = SupervisorAgent(factory, mailbox, api_key="dummy-key")
    
    # Initial state at depth 0
    state = create_initial_state("super", "u1", "s1", Path("/tmp/in"), Path("/tmp/out"))
    state["next_steps"] = ["sub_agent"]
    
    # 1. Test successful spawn
    new_state = supervisor.spawning_node(state)
    assert "Successfully spawned" in new_state["messages"][0]["content"]
    assert new_state["quota"].agent_count == 1
    
    # Check if inbox message was sent
    received = mailbox.receive("sub_agent")
    assert received["sender"] == "super"

def test_supervisor_depth_limit(workspace, mailbox):
    # Setup Factory with depth limit 1
    factory = AgentFactory(workspace, max_spawn_depth=1)
    supervisor = SupervisorAgent(factory, mailbox, api_key="dummy-key")
    
    # Initial state already at depth 1
    state = create_initial_state("super", "u1", "s1", Path("/tmp/in"), Path("/tmp/out"), current_depth=1)
    state["next_steps"] = ["sub_agent"]
    
    # Should fail because parent_depth (1) >= max_spawn_depth (1)
    new_state = supervisor.spawning_node(state)
    assert "Depth exceeded" in new_state["messages"][0]["content"]
