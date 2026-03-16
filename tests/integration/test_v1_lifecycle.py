import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.agents.generator import SystemGeneratorAgent, TaskType
from agent_platform.runtime.agents.validator import SystemValidatorAgent
from agent_platform.runtime.agents.orchestrator import OrchestratorAgent
from agent_platform.runtime.core.dispatcher import ToolDispatcher, ToolRegistry
from agent_platform.runtime.core.schema import ToolSource
from agent_platform.runtime.core.guardrails import GuardrailManager, PolicyGenerator
from agent_platform.runtime.core.sandbox import ProcessSandboxRunner
from agent_platform.runtime.orch.models import PlanningResult, ExecutionStrategy, SubAgentTask, ValidationResult
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def v1_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    
    # Global Resources setup (ISOLATED)
    global_res_dir = root / "global"
    global_res_dir.mkdir(parents=True, exist_ok=True)
    (global_res_dir / "prompts").mkdir(exist_ok=True)
    (global_res_dir / "prompts" / "generator_prompt.txt").write_text("MOCK GENERATOR TEMPLATE")
    (global_res_dir / "guidelines.md").write_text("Safety First")

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "power_user", "sess_v1"
    session_path = initializer.initialize(user_id, session_id)

    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    
    mock_sup_llm = AsyncMock()
    mock_sup_llm.ainvoke.side_effect = [
        PlanningResult(
            thought_process="Mock thought",
            strategy=ExecutionStrategy.DECOMPOSE,
            sub_tasks=[SubAgentTask(agent_id="researcher_1", role=AgentRole.WORKER, instructions="Research")]
        ),
        PlanningResult(
            thought_process="Done.",
            strategy=ExecutionStrategy.FINISH
        )
    ]

    mock_gen_llm = AsyncMock()
    mock_gen_llm.ainvoke.return_value.content = "def researcher_1_func(): return 'mocked'"

    mock_val_llm = AsyncMock()
    mock_val_llm.ainvoke.return_value = ValidationResult(is_valid=True, reasoning="Passed")

    mock_policy_gen = MagicMock(spec=PolicyGenerator)
    mock_policy_gen.generate.return_value = (True, "Allowed")

    generator = SystemGeneratorAgent(llm=mock_gen_llm, workspace=workspace)
    validator = SystemValidatorAgent(llm=mock_val_llm, workspace=workspace)
    orchestrator = OrchestratorAgent(factory, mailbox, generator, llm=mock_sup_llm)
    
    sandbox = ProcessSandboxRunner()
    guardrails = GuardrailManager(policy_generator=mock_policy_gen)
    registry = ToolRegistry(session_path)
    dispatcher = ToolDispatcher(registry, sandbox, guardrails)

    return {
        "env": (user_id, session_id, session_path),
        "orchestrator": orchestrator,
        "dispatcher": dispatcher,
        "generator": generator,
        "validator": validator,
        "mock_val_llm": mock_val_llm
    }

@pytest.mark.asyncio
async def test_v1_full_platform_lifecycle_with_mocks(v1_env):
    user_id, session_id, session_path = v1_env["env"]
    orchestrator = v1_env["orchestrator"]
    dispatcher = v1_env["dispatcher"]
    validator = v1_env["validator"]

    # Correct paths within session
    agent_dir = session_path / "agents" / "super_01"
    inbox = agent_dir / "inbox"
    outbox = agent_dir / "outbox"
    todo = agent_dir / "todo"

    state = create_initial_state(
        "super_01", user_id, session_id, 
        inbox, outbox, 
        todo_path=todo,
        role=AgentRole.SUPERVISOR
    )
    graph = orchestrator.build_graph()
    final_state = await graph.ainvoke(state)
    assert final_state["quota"].agent_count == 1

    # Inject mock generated output for validator to check
    final_state["generated_output"] = "MOCK CODE CONTENT"

    v1_env["mock_val_llm"].ainvoke.return_value = ValidationResult(is_valid=False, reasoning="Violation: destructive")
    val_res = await validator.validate_node(final_state)
    assert val_res["is_valid"] is False

    def search_stub(query, state=None): return f"Search: {query}"
    dispatcher.registry.register_native("google_search", search_stub)
    native_res = await dispatcher.dispatch(final_state, "google_search", query="mock")
    assert native_res["success"] is True
