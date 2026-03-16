import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.guardrails import GuardrailManager, PolicyGenerator
from agent_platform.runtime.core.context_store import FilesystemContextStore
from agent_platform.runtime.orch.state import create_initial_state, AgentRole

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
        "mock_policy_gen": mock_policy_gen
    }

@pytest.mark.asyncio
async def test_guardrail_receives_global_context(guardrail_env):
    """
    Verifies that GuardrailManager fetches facts from the ContextStore 
    and passes them to the PolicyGenerator.
    """
    env = guardrail_env
    agent_id = "supervisor_01"
    
    # 1. Add a fact to the context
    env["context_store"].update_fact(
        agent_id=agent_id,
        fact_id="project_info",
        content="The project root is /tmp/my_project",
        role=AgentRole.SUPERVISOR
    )
    
    # 2. Prepare state
    state = create_initial_state(
        agent_id=agent_id,
        user_id=env["user_id"],
        session_id=env["session_id"],
        inbox_path=env["session_path"] / "agents" / agent_id / "inbox",
        outbox_path=env["session_path"] / "agents" / agent_id / "outbox",
        role=AgentRole.SUPERVISOR
    )
    
    # 3. Trigger validation
    await env["manager"].validate_tool_call(state, "ls", {"path": "/tmp/my_project"})
    
    # 4. Verify mock was called with the context
    args, kwargs = env["mock_policy_gen"].generate.call_args
    visible_context = kwargs.get("visible_context")
    
    assert visible_context is not None
    assert "project_info" in visible_context
    assert "/tmp/my_project" in visible_context

@pytest.mark.asyncio
async def test_guardrail_hierarchy_context_visibility(guardrail_env):
    """
    Verifies that a worker agent sees facts from its parent supervisor 
    during guardrail validation.
    """
    env = guardrail_env
    sup_id = "supervisor_01"
    worker_id = "supervisor_01/worker_01"
    
    # 1. Supervisor updates its context
    env["context_store"].update_fact(
        agent_id=sup_id,
        fact_id="parent_rule",
        content="Implicit access to /data is granted.",
        role=AgentRole.SUPERVISOR
    )
    
    # 2. Worker state (child of supervisor_01)
    state = create_initial_state(
        agent_id=worker_id,
        user_id=env["user_id"],
        session_id=env["session_id"],
        inbox_path=env["session_path"] / "agents" / worker_id / "inbox",
        outbox_path=env["session_path"] / "agents" / worker_id / "outbox",
        role=AgentRole.WORKER
    )
    
    # 3. Trigger validation for WORKER
    await env["manager"].validate_tool_call(state, "read_file", {"path": "/data/config.yaml"})
    
from agent_platform.runtime.agents.supervisor import SupervisorAgent
from agent_platform.runtime.core.mailbox import Mailbox
from agent_platform.runtime.agents.generator import SystemGeneratorAgent

@pytest.mark.asyncio
async def test_supervisor_records_intent_and_results(guardrail_env):
    """
    Verifies that SupervisorAgent uses ContextStore to persist intent 
    and promote sub-agent results.
    """
    env = guardrail_env
    agent_id = "supervisor_01"
    
    # 1. Setup Supervisor with mocks
    mock_factory = MagicMock()
    mock_mailbox = MagicMock()
    mock_generator = MagicMock()
    mock_llm = AsyncMock()
    
    # Mock planning result
    from agent_platform.runtime.orch.models import PlanningResult, ExecutionStrategy
    mock_llm.ainvoke.return_value = PlanningResult(
        thought_process="Planning...",
        strategy=ExecutionStrategy.FINISH
    )
    
    supervisor = SupervisorAgent(
        agent_factory=mock_factory,
        mailbox=mock_mailbox,
        generator=mock_generator,
        llm=mock_llm,
        context_store=env["context_store"]
    )
    
    # 2. Initial state with a user message
    state = create_initial_state(
        agent_id=agent_id,
        user_id=env["user_id"],
        session_id=env["session_id"],
        inbox_path=env["session_path"] / "agents" / agent_id / "inbox",
        outbox_path=env["session_path"] / "agents" / agent_id / "outbox",
        role=AgentRole.SUPERVISOR
    )
    state["messages"] = [{"role": "user", "content": "Analyze the project"}]
    
    # 3. Call planning_node for the first time
    await supervisor.planning_node(state)
    
    # 4. Verify intent was recorded
    intent = env["context_store"].read_fact(agent_id, "initial_intent")
    assert intent is not None
    assert "Analyze the project" in intent

@pytest.mark.asyncio
async def test_generator_receives_global_context(guardrail_env):
    """
    Verifies that SystemGeneratorAgent fetches facts and includes them
    in the human message for prompt generation.
    """
    env = guardrail_env
    agent_id = "supervisor_01"
    
    # 1. Add a fact to the context
    env["context_store"].update_fact(
        agent_id=agent_id,
        fact_id="target_architecture",
        content="The system uses a microservices pattern with FastAPI.",
        role=AgentRole.SUPERVISOR
    )
    
    # 2. Setup Generator with Mock LLM
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value.content = "Generated Prompt"
    
    from agent_platform.runtime.agents.generator import SystemGeneratorAgent
    generator = SystemGeneratorAgent(llm=mock_llm, context_store=env["context_store"])
    
    # 3. Prepare state
    state = create_initial_state(
        agent_id=agent_id,
        user_id=env["user_id"],
        session_id=env["session_id"],
        inbox_path=env["session_path"] / "agents" / agent_id / "inbox",
        outbox_path=env["session_path"] / "agents" / agent_id / "outbox",
        role=AgentRole.SUPERVISOR
    )
    state["next_steps"] = ["researcher_1"]
    state["metadata"]["current_task_instructions"] = "Analyze the codebase."
    
    # 4. Call generator
    await generator.generate_node(state)
    
    # 5. Verify the fact was included in the LLM input
    _, kwargs = mock_llm.ainvoke.call_args
    messages = args[0] if 'args' in locals() else mock_llm.ainvoke.call_args[0][0]
    human_msg = messages[1].content
    
    assert "target_architecture" in human_msg
    assert "microservices pattern" in human_msg
