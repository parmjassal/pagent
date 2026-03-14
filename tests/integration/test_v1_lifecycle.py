import pytest
from pathlib import Path
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.state import create_initial_state
from agent_platform.runtime.generator import SystemGeneratorAgent, TaskType
from agent_platform.runtime.validator import SystemValidatorAgent
from agent_platform.runtime.supervisor import SupervisorAgent
from agent_platform.runtime.dispatcher import ToolDispatcher, ToolSource
from agent_platform.runtime.guardrails import GuardrailManager
from agent_platform.runtime.sandbox import ProcessSandboxRunner
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def v1_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    
    # 1. Setup Global Resources
    (global_prompts := workspace.get_global_dir() / "prompts").mkdir(parents=True)
    (global_prompts / "agent_base.txt").write_text("BASE PROMPT")
    
    # Place guidelines in GLOBAL so they get COPIED during session init
    global_dir = workspace.get_global_dir()
    global_dir.mkdir(parents=True, exist_ok=True)
    (global_dir / "guidelines.md").write_text("# Safety Rules\n- No destructive actions allowed.")

    # 2. Session Initialization (Automated Copying)
    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "power_user", "sess_v1_001"
    session_path = initializer.initialize(user_id, session_id)
    
    # Verification: Ensure guidelines.md was COPIED to user directory
    assert (workspace.get_user_dir(user_id) / "guidelines.md").exists() or (session_path / "guidelines.md").exists()

    # 3. Component Setup
    factory = AgentFactory(workspace, max_spawn_depth=5)
    provider = FilesystemMailboxProvider(session_path)
    mailbox = Mailbox(provider)
    
    sandbox = ProcessSandboxRunner()
    guardrails = GuardrailManager()
    dispatcher = ToolDispatcher(sandbox, guardrails)
    
    generator = SystemGeneratorAgent(llm=None, workspace=workspace)
    validator = SystemValidatorAgent(llm=None, workspace=workspace)
    supervisor = SupervisorAgent(factory, mailbox, generator, api_key="dummy-key")

    return {
        "env": (user_id, session_id, session_path),
        "supervisor": supervisor,
        "dispatcher": dispatcher,
        "generator": generator,
        "validator": validator,
        "mailbox": mailbox
    }

def test_v1_full_platform_lifecycle(v1_env):
    user_id, session_id, session_path = v1_env["env"]
    supervisor = v1_env["supervisor"]
    dispatcher = v1_env["dispatcher"]
    generator = v1_env["generator"]
    validator = v1_env["validator"]
    mailbox = v1_env["mailbox"]

    # --- PHASE 1: ORCHESTRATION ---
    state = create_initial_state("super_01", user_id, session_id, session_path / "agents/super_01/inbox", session_path / "agents/super_01/outbox")
    graph = supervisor.build_graph()
    final_state = graph.invoke(state)
    assert final_state["quota"].agent_count == 1

    # --- PHASE 2: TOOL WRITING & VALIDATION ---
    # Trigger a validation failure by injecting 'delete'
    final_state["generated_output"] = "def unsafe_tool(): import os; os.remove('data.db')"
    
    val_result = validator.validate_node(final_state)
    assert val_result["is_valid"] is False
    assert "destructive" in val_result["validation_feedback"]

    # --- PHASE 3: DISPATCHER ---
    def community_search(query: str): return f"Results for {query}"
    dispatcher.register_tool("google_search", ToolSource.COMMUNITY, func=community_search)
    
    native_res = dispatcher.dispatch(final_state, "google_search", query="test")
    assert native_res["success"] is True

    dispatcher.register_tool("calc_safe", ToolSource.DYNAMIC)
    dynamic_res = dispatcher.dispatch(final_state, "calc_safe", val=10)
    assert dynamic_res["success"] is True
