import pytest
from pathlib import Path
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.state import create_initial_state
from agent_platform.runtime.generator import SystemGeneratorAgent, TaskType
from agent_platform.runtime.validator import SystemValidatorAgent
from agent_platform.runtime.supervisor import SupervisorAgent
from agent_platform.runtime.dispatcher import ToolDispatcher, ToolRegistry, ToolSource
from agent_platform.runtime.guardrails import GuardrailManager
from agent_platform.runtime.sandbox import ProcessSandboxRunner
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def v1_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (global_prompts := workspace.get_global_dir() / "prompts").mkdir(parents=True)
    (global_prompts / "agent_base.txt").write_text("BASE PROMPT")
    (workspace.get_global_dir() / "guidelines.md").write_text("# Safety\n- No destructive")

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "power_user", "sess_v1_001"
    session_path = initializer.initialize(user_id, session_id)

    factory = AgentFactory(workspace, max_spawn_depth=5)
    provider = FilesystemMailboxProvider(session_path)
    mailbox = Mailbox(provider)
    
    # Refined Registry & Dispatcher
    registry = ToolRegistry(session_path)
    sandbox = ProcessSandboxRunner()
    guardrails = GuardrailManager()
    dispatcher = ToolDispatcher(registry, sandbox, guardrails)
    
    def community_search(query: str): return f"Results for {query}"
    dispatcher.registry.register_native("google_search", community_search)

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

    # 1. Orchestration
    state = create_initial_state("super_01", user_id, session_id, session_path / "agents/super_01/inbox", session_path / "agents/super_01/outbox")
    graph = supervisor.build_graph()
    final_state = graph.invoke(state)

    # 2. Validation Failure (Simulated)
    final_state["generated_output"] = "def unsafe(): import os; os.remove('db')"
    val_result = validator.validate_node(final_state)
    assert val_result["is_valid"] is False

    # 3. Native Execution
    native_res = dispatcher.dispatch(final_state, "google_search", query="test")
    assert native_res["success"] is True
    assert native_res["source"] == "native"

    # 4. Dynamic Execution & Persistence
    dispatcher.registry.register_dynamic("calc_risk")
    dynamic_res = dispatcher.dispatch(final_state, "calc_risk", val=5)
    assert dynamic_res["success"] is True
    assert dynamic_res["source"] == "sandbox"
    
    # Check Persistence
    assert (session_path / "tool_registry.json").exists()
