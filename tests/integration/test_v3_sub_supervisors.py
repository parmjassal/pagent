import pytest
from pathlib import Path
from agent_platform.runtime.state import create_initial_state, AgentRole
from agent_platform.runtime.quota import update_quota
from agent_platform.runtime.supervisor import SupervisorAgent
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def v3_env(tmp_path):
    workspace = WorkspaceContext(root=tmp_path / ".pagent")
    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(tmp_path / "sess"))
    supervisor = SupervisorAgent(factory, mailbox, None, api_key="dummy")
    return {"supervisor": supervisor, "workspace": workspace, "mailbox": mailbox}

def test_sub_supervisor_spawning_flow(v3_env):
    """Verifies Top-Level Super can spawn a Sub-Level Super."""
    env = v3_env
    supervisor = env["supervisor"]
    
    # 1. Top Level Role = SUPERVISOR
    state = create_initial_state("top_super", "u1", "s1", Path("/tmp"), Path("/tmp"), role=AgentRole.SUPERVISOR)
    
    # 2. Decompose into Sub-Supervisor
    # Our mocked decomposition node returns 'metadata': {'next_agent_role': AgentRole.SUPERVISOR}
    decomp_res = supervisor.task_decomposition_node(state)
    state.update(decomp_res)
    
    assert state["metadata"]["next_agent_role"] == AgentRole.SUPERVISOR
    
    # 3. Spawn the Sub-Supervisor
    spawn_res = supervisor.spawning_node(state)
    
    assert "Successfully spawned supervisor" in spawn_res["messages"][-1]["content"]
    
    # 4. Verify Mailbox for Sub-Supervisor includes the correct Role
    msg = env["mailbox"].receive("sub_supervisor_01")
    assert msg["role"] == AgentRole.SUPERVISOR
    
    # 5. Check if Sub-Supervisor Directory exists
    sub_super_dir = env["workspace"].get_agent_dir("u1", "s1", "sub_supervisor_01")
    assert sub_super_dir.exists()
