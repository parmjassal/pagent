import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.orch.quota import update_quota, SessionQuota
from agent_platform.runtime.agents.generator import SystemGeneratorAgent
from agent_platform.runtime.agents.orchestrator import OrchestratorAgent
from agent_platform.runtime.orch.unit_compiler import UnitCompiler
from agent_platform.runtime.orch.result_hook import OffloadingResultHook
from agent_platform.runtime.orch.models import PlanningResult, ExecutionStrategy, SubAgentTask
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
    mock_llm.ainvoke.return_value = AIMessage(content=PlanningResult(
        thought_process="Spawning worker for unit test.",
        strategy=ExecutionStrategy.DECOMPOSE,
        sub_tasks=[SubAgentTask(agent_id="worker_1", role=AgentRole.WORKER, instructions="Do work")]
    ).model_dump_json())

    generator = SystemGeneratorAgent(llm=AsyncMock(), workspace=workspace)
    # Mock generator for executor_node
    generator.generate_node = AsyncMock(return_value={"generated_output": "Process task."})
    
    dispatcher = MagicMock(spec=ToolDispatcher)
    compiler = UnitCompiler(factory, mailbox, generator, dispatcher, result_hook)
    
    supervisor = OrchestratorAgent(
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
    Verifies that Orchestrator can trigger a Worker subgraph 
    and merge the result back.
    """
    env = v7_env
    supervisor = env["supervisor"]
    
    agent_dir = env["session_path"] / "agents" / "super"
    agent_dir.mkdir(parents=True, exist_ok=True)
    inbox = agent_dir / "inbox"
    outbox = agent_dir / "outbox"
    todo = agent_dir / "todo"
    
    state = create_initial_state(
        "super", env["user_id"], env["session_id"], 
        inbox, outbox, 
        todo_path=todo,
        role=AgentRole.SUPERVISOR
    )
    
    # 1. PLAN
    decomp_res = await supervisor.planner_node(state)
    state.update(decomp_res)

    # 1.5 DISPATCH
    dispatch_res = await supervisor.dispatcher_node(state)
    state.update(dispatch_res)

    # 2. EXECUTOR
    mock_worker_graph = MagicMock()
    mock_worker_graph.ainvoke = AsyncMock()
    mock_worker_graph.ainvoke.return_value = {
        "final_result": {"content": "Success from worker_1"},
        "quota": SessionQuota(agent_count=1), 
        "messages": [{"role": "assistant", "content": "Worker finished."}],
        "next_steps": []
    }
    
    env["compiler"].compile_unit = MagicMock(return_value=mock_worker_graph)
    
    spawn_res = await supervisor.executor_node(state)
    
    # 3. VERIFY RESULT in metadata
    assert "Success from worker_1" in str(spawn_res["metadata"]["task_result"])
    assert spawn_res["quota"].agent_count == 1
