import pytest
from pathlib import Path
from langchain_openai import ChatOpenAI
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.lifecycle import AgentLifecycleManager
from agent_platform.runtime.quota import SessionQuota
from agent_platform.runtime.state import create_initial_state
from agent_platform.runtime.prompt_writer import DynamicPromptWriter
from agent_platform.runtime.supervisor import SupervisorAgent
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def integ_env(tmp_path):
    """
    Sets up a complete integration environment in a temporary directory.
    - .pagent root
    - Global resources
    - Session context
    """
    # 1. Workspace & Resources
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    
    global_dir = workspace.get_global_dir()
    (global_dir / "prompts").mkdir(parents=True)
    (global_dir / "prompts" / "agent_base.txt").write_text("BASE TEMPLATE: Be helpful and concise.")

    # 2. Initialize Session
    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "test_user", "integ_sess_001"
    session_path = initializer.initialize(user_id, session_id)

    # 3. Core Components
    factory = AgentFactory(workspace, max_spawn_depth=5)
    lifecycle = AgentLifecycleManager(workspace, factory)
    
    provider = FilesystemMailboxProvider(session_path)
    mailbox = Mailbox(provider)
    
    # Using a dummy-key to avoid real LLM calls during flow testing
    prompt_writer = DynamicPromptWriter(llm=None, workspace=workspace)
    supervisor = SupervisorAgent(factory, mailbox, prompt_writer, api_key="dummy-key")

    return {
        "workspace": workspace,
        "session_path": session_path,
        "user_id": user_id,
        "session_id": session_id,
        "supervisor": supervisor,
        "mailbox": mailbox
    }

def test_v0_orchestration_flow(integ_env):
    """
    Verifies the full 'v0' orchestration loop:
    Decompose -> Generate Prompt -> Spawn Sub-agent -> Mailbox Handover
    """
    env = integ_env
    supervisor = env["supervisor"]
    
    # 1. Create Initial State for Supervisor
    # Supervisor is at depth 0
    initial_state = create_initial_state(
        agent_id="supervisor_01",
        user_id=env["user_id"],
        session_id=env["session_id"],
        inbox_path=env["session_path"] / "agents" / "supervisor_01" / "inbox",
        outbox_path=env["session_path"] / "agents" / "supervisor_01" / "outbox",
        max_agents=10
    )
    
    # 2. Run the Graph
    graph = supervisor.build_graph()
    # Execute the full graph synchronously
    final_state = graph.invoke(initial_state)

    # 3. VERIFICATIONS
    
    # A. Check State Transitions
    assert "researcher_1" not in final_state["next_steps"] # Should be cleared after spawning
    assert final_state["quota"].agent_count == 1 # 1 agent spawned
    assert "Successfully spawned researcher_1" in final_state["messages"][-1]["content"]

    # B. Check Filesystem (Agent Context)
    researcher_dir = env["session_path"] / "agents" / "researcher_1"
    assert researcher_dir.exists()
    assert (researcher_dir / "inbox").exists()

    # C. Check Mailbox (Message Delivery)
    received_msg = env["mailbox"].receive("researcher_1")
    assert received_msg is not None
    assert received_msg["sender"] == "supervisor_01"
    assert "system_prompt" in received_msg
    assert "BASE TEMPLATE: Be helpful and concise." in received_msg["system_prompt"]
    assert "Task decomposition" in received_msg["system_prompt"]

    # D. Verify Snapshot Engine (Manual Check for now)
    # Ensuring the snapshot directory was created during initialization
    assert (env["session_path"] / "snapshot").exists()
