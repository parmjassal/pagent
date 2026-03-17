import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.guardrails import GuardrailManager, PolicyGenerator
from agent_platform.runtime.core.context_store import FilesystemContextStore
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.agents.orchestrator import OrchestratorAgent
from agent_platform.runtime.core.mailbox import Mailbox
from agent_platform.runtime.agents.generator import SystemGeneratorAgent
from agent_platform.runtime.orch.models import PlanningResult, ExecutionStrategy

@pytest.fixture
def guardrail_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    
    user_id, session_id = "test_user", "sess_guard_v12"
    session_path = workspace.ensure_session_structure(user_id, session_id)
    
    context_store = FilesystemContextStore(session_path)
    
    mock_policy_gen = AsyncMock(spec=PolicyGenerator)
    # Default to allowed
    mock_policy_gen.generate.return_value = (True, "Allowed by default")
    
    manager = GuardrailManager(policy_generator=mock_policy_gen, context_store=context_store)
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "session_path": session_path,
        "context_store": context_store,
        "manager": manager,
        "mock_policy_gen": mock_policy_gen,
        "workspace": workspace
    }

@pytest.mark.asyncio
async def test_guardrail_receives_global_context(guardrail_env):
    env = guardrail_env
    agent_id = "supervisor_01"
    
    env["context_store"].update_fact(
        agent_id=agent_id,
        fact_id="project_info",
        content="The project root is /tmp/my_project",
        role=AgentRole.SUPERVISOR
    )
    
    state = create_initial_state(
        agent_id=agent_id,
        user_id=env["user_id"],
        session_id=env["session_id"],
        inbox_path=env["session_path"] / "agents" / agent_id / "inbox",
        outbox_path=env["session_path"] / "agents" / agent_id / "outbox",
        role=AgentRole.SUPERVISOR
    )
    
    await env["manager"].validate_tool_call(state, "ls", {"path": "/tmp/my_project"})
    
    _, kwargs = env["mock_policy_gen"].generate.call_args
    visible_context = kwargs.get("visible_context")
    assert "project_info" in visible_context
    assert "/tmp/my_project" in visible_context

@pytest.mark.asyncio
async def test_guardrail_hierarchy_context_visibility(guardrail_env):
    env = guardrail_env
    sup_id = "supervisor_01"
    worker_id = "supervisor_01/worker_01"
    
    env["context_store"].update_fact(
        agent_id=sup_id,
        fact_id="parent_rule",
        content="Implicit access to /data is granted.",
        role=AgentRole.SUPERVISOR
    )
    
    state = create_initial_state(
        agent_id=worker_id,
        user_id=env["user_id"],
        session_id=env["session_id"],
        inbox_path=env["session_path"] / "agents" / worker_id / "inbox",
        outbox_path=env["session_path"] / "agents" / worker_id / "outbox",
        role=AgentRole.WORKER
    )
    
    await env["manager"].validate_tool_call(state, "read_file", {"path": "/data/config.yaml"})
    
    _, kwargs = env["mock_policy_gen"].generate.call_args
    visible_context = kwargs.get("visible_context")
    assert "parent_rule" in visible_context
    assert "Implicit access to /data" in visible_context

@pytest.mark.asyncio
async def test_orchestrator_planning_manifest_injection(guardrail_env):
    """Verifies that OrchestratorAgent injects the tool manifest into system prompt during planning."""
    env = guardrail_env
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content=PlanningResult(thought_process="...", strategy=ExecutionStrategy.FINISH).model_dump_json())
    
    manifest = "## Available Tools\n- **ls**: list files"
    orchestrator = OrchestratorAgent(
        agent_factory=MagicMock(),
        mailbox=MagicMock(),
        generator=MagicMock(),
        llm=mock_llm,
        tool_manifest=manifest
    )
    
    agent_dir = env["session_path"] / "agents" / "sup"
    agent_dir.mkdir(parents=True, exist_ok=True)
    state = create_initial_state("sup", "u", "s", Path("/tmp"), Path("/tmp"), todo_path=agent_dir/"todo", role=AgentRole.SUPERVISOR)
    await orchestrator.planner_node(state)
    
    call_args = mock_llm.ainvoke.call_args[0][0]
    system_msg = call_args[0].content
    assert "## Available Tools" in system_msg
    assert "ls" in system_msg

@pytest.mark.asyncio
async def test_agent_invokes_correct_tool_name(guardrail_env):
    env = guardrail_env
    from agent_platform.runtime.core.dispatcher import ToolDispatcher, ToolRegistry
    from agent_platform.runtime.orch.tool_node import AgentToolNode
    
    registry = ToolRegistry(env["session_path"])
    registry.register_native("ls", lambda path, state=None: ["file1.txt"], summary="List files")
    
    mock_dispatcher = MagicMock(spec=ToolDispatcher)
    mock_dispatcher.dispatch = AsyncMock(return_value={"success": True, "output": "file1.txt", "source": "native"})
    
    tool_node = AgentToolNode(mock_dispatcher)
    
    state = create_initial_state("wrk", "u", "s", Path("/tmp"), Path("/tmp"), role=AgentRole.WORKER)
    state["metadata"]["next_tool_call"] = {"name": "ls", "args": {"path": "."}, "id": "call_123"}
    
    await tool_node(state)
    
    args, _ = mock_dispatcher.dispatch.call_args
    assert args[1] == "ls"
