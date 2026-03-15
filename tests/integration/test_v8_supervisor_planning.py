import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.agents.supervisor import SupervisorAgent
from agent_platform.runtime.orch.models import PlanningResult, ExecutionStrategy, SubAgentTask, ToolCall
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def plan_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "planner", "sess_plan"
    session_path = initializer.initialize(user_id, session_id)

    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    
    mock_llm = AsyncMock()
    supervisor = SupervisorAgent(factory, mailbox, MagicMock(), llm=mock_llm)

    return {"supervisor": supervisor, "mock_llm": mock_llm, "user_id": user_id, "session_id": session_id}

@pytest.mark.asyncio
async def test_supervisor_planning_decompose(plan_env):
    """Verifies that supervisor chooses DECOMPOSE when LLM signals it."""
    sup = plan_env["supervisor"]
    
    # 1. Mock DECOMPOSE strategy
    plan_env["mock_llm"].ainvoke.return_value = PlanningResult(
        thought_process="Complex task, need help.",
        strategy=ExecutionStrategy.DECOMPOSE,
        sub_tasks=[SubAgentTask(agent_id="helper", role=AgentRole.WORKER, instructions="Task")]
    )

    state = create_initial_state("s1", plan_env["user_id"], plan_env["session_id"], Path("/tmp"), Path("/tmp"))
    
    # 2. Run Planning Node
    res = await sup.planning_node(state)
    
    assert res["metadata"]["strategy"] == ExecutionStrategy.DECOMPOSE
    assert "helper" in res["next_steps"]
    assert sup._should_continue(res) == "generate_prompt"

@pytest.mark.asyncio
async def test_supervisor_planning_tool_use(plan_env):
    """Verifies that supervisor chooses TOOL_USE for simple tasks."""
    sup = plan_env["supervisor"]
    
    # 1. Mock TOOL_USE strategy
    plan_env["mock_llm"].ainvoke.return_value = PlanningResult(
        thought_process="Simple task, I'll do it myself.",
        strategy=ExecutionStrategy.TOOL_USE,
        tool_call=ToolCall(name="ls", args={"path": "."})
    )

    state = create_initial_state("s1", plan_env["user_id"], plan_env["session_id"], Path("/tmp"), Path("/tmp"))
    
    # 2. Run Planning Node
    res = await sup.planning_node(state)
    
    assert res["metadata"]["strategy"] == ExecutionStrategy.TOOL_USE
    assert res["metadata"]["next_tool_call"]["name"] == "ls"
    assert sup._should_continue(res) == "tools"
