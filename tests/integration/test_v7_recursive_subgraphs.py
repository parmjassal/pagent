import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.orch.quota import update_quota, SessionQuota
from agent_platform.runtime.agents.generator import SystemGeneratorAgent
from agent_platform.runtime.agents.supervisor import SupervisorAgent
from agent_platform.runtime.orch.unit_compiler import UnitCompiler
from agent_platform.runtime.orch.result_hook import OffloadingResultHook
from agent_platform.runtime.orch.models import DecompositionResult, SubAgentTask
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider
from agent_platform.runtime.core.dispatcher import ToolDispatcher, ToolRegistry

@pytest.fixture
def v7_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "lego_user", "sess_v7"
    session_path = initializer.initialize(user_id, session_id)

    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    result_hook = OffloadingResultHook(session_path / "knowledge")
    
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = DecompositionResult(
        thought_process="Spawning worker for unit test.",
        sub_tasks=[SubAgentTask(agent_id="worker_1", role=AgentRole.WORKER, instructions="Do work")]
    )

    generator = SystemGeneratorAgent(llm=AsyncMock(), workspace=workspace)
    dispatcher = MagicMock(spec=ToolDispatcher)
    compiler = UnitCompiler(factory, mailbox, generator, dispatcher, result_hook)
    
    supervisor = SupervisorAgent(
        factory, mailbox, generator, 
        llm=mock_llm, 
        unit_compiler=compiler
    )

    return {
        "user_id": user_id, "session_id": session_id, "session_path": session_path,
        "supervisor": supervisor, "compiler": compiler, "mock_llm": mock_llm
    }

@pytest.mark.asyncio
async def test_v7_recursive_subgraph_invocation(v7_env):
    """
    Verifies that Supervisor can trigger a Worker subgraph 
    and merge the result back.
    """
    env = v7_env
    supervisor = env["supervisor"]
    
    state = create_initial_state("super", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
    
    # 1. DECOMPOSE
    decomp_res = await supervisor.task_decomposition_node(state)
    state.update(decomp_res)

    # 2. SPAWN
    mock_worker_graph = MagicMock()
    mock_worker_graph.ainvoke = AsyncMock()
    mock_worker_graph.ainvoke.return_value = {
        "final_result": {"content": "Success from worker_1"},
        "quota": SessionQuota(agent_count=1), 
        "messages": [{"role": "assistant", "content": "Worker finished."}],
        "next_steps": []
    }
    
    env["compiler"].compile_unit = MagicMock(return_value=mock_worker_graph)
    
    spawn_res = await supervisor.spawning_node(state)
    
    # 3. VERIFY MERGE
    assert "Sub-agent worker_1 returned" in spawn_res["messages"][0]["content"]
    assert "Success from worker_1" in spawn_res["messages"][0]["content"]
    assert len(spawn_res["next_steps"]) == 0
    assert spawn_res["quota"].agent_count == 1
