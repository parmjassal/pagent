import pytest
from pathlib import Path
from unittest.mock import MagicMock
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.orch.quota import update_quota
from agent_platform.runtime.agents.supervisor import SupervisorAgent
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.orch.models import DecompositionResult, SubAgentTask
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def v3_env(tmp_path):
    workspace = WorkspaceContext(root=tmp_path / ".pagent")
    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(tmp_path / "sess"))
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = DecompositionResult(
        thought_process="Spawning sub-super",
        sub_tasks=[SubAgentTask(agent_id="sub_supervisor_01", role=AgentRole.SUPERVISOR, instructions="Task")]
    )

    supervisor = SupervisorAgent(factory, mailbox, None, llm=mock_llm)
    return {"supervisor": supervisor, "workspace": workspace, "mailbox": mailbox}

def test_sub_supervisor_spawning_flow(v3_env):
    env = v3_env
    supervisor = env["supervisor"]
    
    state = create_initial_state("top_super", "u1", "s1", Path("/tmp"), Path("/tmp"), role=AgentRole.SUPERVISOR)
    
    decomp_res = supervisor.task_decomposition_node(state)
    state.update(decomp_res)
    
    assert state["metadata"]["next_agent_role"] == AgentRole.SUPERVISOR
    
    spawn_res = supervisor.spawning_node(state)
    assert "Successfully spawned supervisor" in spawn_res["messages"][-1]["content"]
    
    msg = env["mailbox"].receive("sub_supervisor_01")
    assert msg["role"] == AgentRole.SUPERVISOR
