import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.orch.quota import update_quota
from agent_platform.runtime.agents.orchestrator import OrchestratorAgent
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
    mock_llm.ainvoke.return_value = AIMessage(content=PlanningResult(
        thought_process="Spawning sub-super",
        strategy=ExecutionStrategy.DECOMPOSE,
        sub_tasks=[SubAgentTask(agent_id="sub_supervisor_01", role=AgentRole.SUPERVISOR, instructions="Task")]
    ).model_dump_json())

    # Mock generator to return a dict with generated_output
    mock_generator = AsyncMock()
    mock_generator.generate_node.return_value = {"generated_output": "Process task prompt."}

    supervisor = OrchestratorAgent(factory, mailbox, mock_generator, llm=mock_llm)
    return {"supervisor": supervisor, "workspace": workspace, "mailbox": mailbox}

@pytest.mark.asyncio
async def test_sub_supervisor_spawning_flow(v3_env):
    env = v3_env
    supervisor = env["supervisor"]
    
    # Use isolated todo_path in session
    session_path = env["workspace"].get_session_dir("u1", "s1")
    todo_path = session_path / "agents" / "top_super" / "todo"
    todo_path.mkdir(parents=True, exist_ok=True)

    state = create_initial_state(
        "top_super", "u1", "s1", 
        Path("/tmp"), Path("/tmp"), 
        todo_path=todo_path,
        role=AgentRole.SUPERVISOR
    )
    
    decomp_res = await supervisor.planner_node(state)
    state.update(decomp_res)
    
    assert state["metadata"]["strategy"] == ExecutionStrategy.DECOMPOSE
    
    # Dispatch to set up next_task
    dispatch_res = await supervisor.dispatcher_node(state)
    state.update(dispatch_res)

    spawn_res = await supervisor.executor_node(state)
    # Orchestrator doesn't return messages directly in executor_node for async mailbox, it returns metadata and quota.
    # Actually OrchestratorAgent.executor_node returns:
    # return {
    #     "metadata": {"task_result": f"Spawned {sub_agent_id} via Mailbox"},
    #     "quota": SessionQuota(agent_count=1)
    # }
    assert "Spawned" in spawn_res["metadata"]["task_result"]
    assert "sub_supervisor_01" in spawn_res["metadata"]["task_result"]
    
    # Verify Mailbox Role
    msg = env["mailbox"].receive("top_super/sub_supervisor_01")
    assert msg["role"] == AgentRole.SUPERVISOR
