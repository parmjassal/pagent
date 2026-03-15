import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.orch.quota import update_quota
from agent_platform.runtime.agents.supervisor import SupervisorAgent
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.orch.models import PlanningResult, ExecutionStrategy, SubAgentTask
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def v3_env(tmp_path):
    workspace = WorkspaceContext(root=tmp_path / ".pagent")
    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(tmp_path / "sess"))
    
    # MOCK LLM ASYNC
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = PlanningResult(
        thought_process="Spawning sub-super",
        strategy=ExecutionStrategy.DECOMPOSE,
        sub_tasks=[SubAgentTask(agent_id="sub_supervisor_01", role=AgentRole.SUPERVISOR, instructions="Task")]
    )

    supervisor = SupervisorAgent(factory, mailbox, AsyncMock(), llm=mock_llm)
    return {"supervisor": supervisor, "workspace": workspace, "mailbox": mailbox}

@pytest.mark.asyncio
async def test_sub_supervisor_spawning_flow(v3_env):
    env = v3_env
    supervisor = env["supervisor"]
    
    state = create_initial_state("top_super", "u1", "s1", Path("/tmp"), Path("/tmp"), role=AgentRole.SUPERVISOR)
    
    decomp_res = await supervisor.planning_node(state)
    state.update(decomp_res)
    
    assert state["metadata"]["strategy"] == ExecutionStrategy.DECOMPOSE
    
    spawn_res = await supervisor.spawning_node(state)
    assert "Spawned top_super/sub_supervisor_01 via Mailbox" in spawn_res["messages"][-1]["content"]
    
    # Verify Mailbox Role
    msg = env["mailbox"].receive("top_super/sub_supervisor_01")
    assert msg["role"] == AgentRole.SUPERVISOR
