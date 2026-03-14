import pytest
from pathlib import Path
from unittest.mock import MagicMock
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state
from agent_platform.runtime.agents.generator import SystemGeneratorAgent
from agent_platform.runtime.agents.supervisor import SupervisorAgent
from agent_platform.runtime.orch.models import DecompositionResult, SubAgentTask
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def integ_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "test_user", "sess_v0"
    session_path = initializer.initialize(user_id, session_id)

    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    
    mock_sup_llm = MagicMock()
    mock_sup_llm.invoke.return_value = DecompositionResult(
        thought_process="Decomposing...",
        sub_tasks=[SubAgentTask(agent_id="researcher_1", role="worker", instructions="Task")]
    )
    mock_gen_llm = MagicMock()
    mock_gen_llm.invoke.return_value.content = "SYSTEM PROMPT"

    generator = SystemGeneratorAgent(llm=mock_gen_llm, workspace=workspace)
    supervisor = SupervisorAgent(factory, mailbox, generator, llm=mock_sup_llm)

    return {
        "user_id": user_id, "session_id": session_id, "session_path": session_path,
        "supervisor": supervisor, "mailbox": mailbox
    }

def test_v0_orchestration_flow(integ_env):
    env = integ_env
    supervisor = env["supervisor"]
    
    initial_state = create_initial_state("super", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
    graph = supervisor.build_graph()
    final_state = graph.invoke(initial_state)

    assert final_state["quota"].agent_count == 1
    assert (env["session_path"] / "agents" / "researcher_1").exists()
