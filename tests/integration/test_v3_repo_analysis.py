import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.orch.quota import update_quota
from agent_platform.runtime.agents.generator import SystemGeneratorAgent
from agent_platform.runtime.agents.orchestrator import OrchestratorAgent
from agent_platform.runtime.agents.search_agent import SemanticSearchAgent
from agent_platform.runtime.orch.models import PlanningResult, ExecutionStrategy, SubAgentTask
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def repo_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)
    
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()
    (repo_dir / "raft.py").write_text("Raft Consensus")

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "repo_expert", "sess_repo_v3"
    session_path = initializer.initialize(user_id, session_id)

    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = PlanningResult(
        thought_process="Decomposing for repo analysis.",
        strategy=ExecutionStrategy.DECOMPOSE,
        sub_tasks=[SubAgentTask(agent_id="analyst_agent", role=AgentRole.WORKER, instructions="Index and query.")]
    )

    mock_gen_llm = AsyncMock()
    mock_gen_llm.ainvoke.return_value.content = "SYSTEM PROMPT"
    
    generator = SystemGeneratorAgent(llm=mock_gen_llm, workspace=workspace)
    orchestrator = OrchestratorAgent(factory, mailbox, generator, llm=mock_llm)
    search_agent = SemanticSearchAgent(workspace)

    return {
        "user_id": user_id, "session_id": session_id, "session_path": session_path,
        "repo_path": repo_dir, "orchestrator": orchestrator, "search_agent": search_agent,
        "mailbox": mailbox, "mock_llm": mock_llm
    }

@pytest.mark.asyncio
async def test_v3_repo_analysis_with_mocked_llm(repo_env):
    env = repo_env
    orchestrator = env["orchestrator"]
    search_agent = env["search_agent"]
    
    user_id, session_id = env["user_id"], env["session_id"]
    agent_dir = env["session_path"] / "agents" / "super_01"
    inbox, outbox, todo = agent_dir/"inbox", agent_dir/"outbox", agent_dir/"todo"
    
    state = create_initial_state(
        "super_01", user_id, session_id, 
        inbox, outbox, 
        todo_path=todo,
        role=AgentRole.SUPERVISOR
    )
    
    # 1. Plan
    plan_res = await orchestrator.planner_node(state)
    state.update(plan_res)
    env["mock_llm"].ainvoke.assert_called_once()
    
    # 2. Dispatch
    disp_res = await orchestrator.dispatcher_node(state)
    state.update(disp_res)
    assert state["metadata"]["next_task"]["assigned_to"] == "analyst_agent"

    # 3. Execute (Spawning)
    exec_res = await orchestrator.executor_node(state)
    assert "Spawned super_01/analyst_agent via Mailbox" in exec_res["metadata"]["task_result"]

    analyst_state = create_initial_state("analyst_agent", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
    analyst_state["metadata"]["target_folder"] = str(env["repo_path"])
    index_res = search_agent.index_node(analyst_state)
    assert index_res["metadata"]["index_ready"] is True
