import pytest
from pathlib import Path
from unittest.mock import MagicMock
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.state import create_initial_state
from agent_platform.runtime.generator import SystemGeneratorAgent, TaskType
from agent_platform.runtime.validator import SystemValidatorAgent
from agent_platform.runtime.supervisor import SupervisorAgent
from agent_platform.runtime.dispatcher import ToolDispatcher, ToolRegistry, ToolSource
from agent_platform.runtime.guardrails import GuardrailManager, PolicyGenerator
from agent_platform.runtime.sandbox import ProcessSandboxRunner
from agent_platform.runtime.models import DecompositionResult, SubAgentTask, ValidationResult
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def v1_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)
    (workspace.get_global_dir() / "guidelines.md").write_text("Safety First")

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "power_user", "sess_v1"
    session_path = initializer.initialize(user_id, session_id)

    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    
    # 1. SETUP MOCKS
    mock_sup_llm = MagicMock()
    mock_sup_llm.invoke.return_value = DecompositionResult(
        thought_process="Mock thought",
        sub_tasks=[SubAgentTask(agent_id="researcher_1", role="worker", instructions="Research")]
    )

    mock_gen_llm = MagicMock()
    mock_gen_llm.invoke.return_value.content = "def researcher_1_func(): return 'mocked'"

    mock_val_llm = MagicMock()
    # Initial success response
    mock_val_llm.invoke.return_value = ValidationResult(is_valid=True, reasoning="Passed")

    mock_policy_gen = MagicMock(spec=PolicyGenerator)
    mock_policy_gen.generate.return_value = (True, "Allowed")

    # 2. INJECT MOCKS
    generator = SystemGeneratorAgent(llm=mock_gen_llm, workspace=workspace)
    validator = SystemValidatorAgent(llm=mock_val_llm, workspace=workspace)
    supervisor = SupervisorAgent(factory, mailbox, generator, llm=mock_sup_llm)
    
    sandbox = ProcessSandboxRunner()
    guardrails = GuardrailManager(policy_generator=mock_policy_gen)
    registry = ToolRegistry(session_path)
    dispatcher = ToolDispatcher(registry, sandbox, guardrails)

    return {
        "env": (user_id, session_id, session_path),
        "supervisor": supervisor,
        "dispatcher": dispatcher,
        "generator": generator,
        "validator": validator,
        "mock_val_llm": mock_val_llm
    }

def test_v1_full_platform_lifecycle_with_mocks(v1_env):
    user_id, session_id, session_path = v1_env["env"]
    supervisor = v1_env["supervisor"]
    dispatcher = v1_env["dispatcher"]
    validator = v1_env["validator"]

    # 1. ORCHESTRATION
    state = create_initial_state("super_01", user_id, session_id, Path("/tmp"), Path("/tmp"))
    graph = supervisor.build_graph()
    final_state = graph.invoke(state)
    assert final_state["quota"].agent_count == 1

    # 2. VALIDATION FAILURE CASE
    # Change mock behavior for negative test
    v1_env["mock_val_llm"].invoke.return_value = ValidationResult(is_valid=False, reasoning="Violation: destructive")
    val_res = validator.validate_node(final_state)
    assert val_res["is_valid"] is False

    # 3. NATIVE EXECUTION
    def search_stub(query): return f"Search: {query}"
    dispatcher.registry.register_native("google_search", search_stub)
    native_res = dispatcher.dispatch(final_state, "google_search", query="mock")
    assert native_res["success"] is True
