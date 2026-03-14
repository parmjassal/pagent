import pytest
from pathlib import Path
from unittest.mock import MagicMock
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.state import create_initial_state, AgentRole
from agent_platform.runtime.quota import update_quota
from agent_platform.runtime.generator import SystemGeneratorAgent
from agent_platform.runtime.supervisor import SupervisorAgent
from agent_platform.runtime.search_agent import SemanticSearchAgent
from agent_platform.runtime.models import DecompositionResult, SubAgentTask
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

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
    
    # 1. Setup Mock LLM for injection
    mock_llm = MagicMock()
    # Configure it to return a structured DecompositionResult
    mock_llm.invoke.return_value = DecompositionResult(
        thought_process="Decomposing for repo analysis.",
        sub_tasks=[SubAgentTask(agent_id="analyst_agent", role=AgentRole.WORKER, instructions="Index and query.")]
    )

    generator = SystemGeneratorAgent(llm=None, workspace=workspace)
    # 2. Inject the Mock LLM into the Supervisor
    supervisor = SupervisorAgent(factory, mailbox, generator, llm=mock_llm)
    search_agent = SemanticSearchAgent(workspace)

    return {
        "user_id": user_id, "session_id": session_id, "session_path": session_path,
        "repo_path": repo_dir, "supervisor": supervisor, "search_agent": search_agent,
        "mailbox": mailbox, "mock_llm": mock_llm
    }

def test_v3_repo_analysis_with_mocked_llm(repo_env):
    env = repo_env
    supervisor = env["supervisor"]
    search_agent = env["search_agent"]
    
    # Initial state
    state = create_initial_state("super_01", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
    
    # 1. ORCHESTRATION (Triggers the Mock LLM)
    # The decompose node now calls env["mock_llm"].invoke()
    decomp_state = supervisor.task_decomposition_node(state)
    assert "analyst_agent" in decomp_state["next_steps"]
    env["mock_llm"].invoke.assert_called_once()

    # 2. SPAWNING
    state.update(decomp_state)
    spawn_res = supervisor.spawning_node(state)
    assert "Successfully spawned worker: analyst_agent" in spawn_res["messages"][-1]["content"]

    # 3. INDEXING (Using real Search Agent)
    analyst_state = create_initial_state("analyst_agent", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
    analyst_state["metadata"]["target_folder"] = str(env["repo_path"])
    index_res = search_agent.index_node(analyst_state)
    assert index_res["metadata"]["index_ready"] is True
