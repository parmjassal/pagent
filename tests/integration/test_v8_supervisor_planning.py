import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.agents.orchestrator import OrchestratorAgent
from agent_platform.runtime.orch.models import PlanningResult, ExecutionStrategy, SubAgentTask, Action
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider
from agent_platform.runtime.core.todo import TODOManager, TaskType

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
    supervisor = OrchestratorAgent(factory, mailbox, MagicMock(), llm=mock_llm)

    return {
        "supervisor": supervisor, "mock_llm": mock_llm, 
        "user_id": user_id, "session_id": session_id,
        "session_path": session_path
    }

from langchain_core.messages import AIMessage

@pytest.mark.asyncio
async def test_supervisor_planning_decompose(plan_env):
    """Verifies that supervisor chooses DECOMPOSE when LLM signals it."""
    sup = plan_env["supervisor"]
    
    # 1. Mock DECOMPOSE strategy
    result_obj = PlanningResult(
        thought_process="Complex task, need help.",
        action_sequence=[
            Action(strategy=ExecutionStrategy.DECOMPOSE, sub_tasks=[
                SubAgentTask(agent_id="new_helper", role=AgentRole.WORKER, instructions="Task")
            ])
        ]
    )
    # The initial call to self.llm.ainvoke in planner_node expects an object with a .content attribute.
    plan_env["mock_llm"].ainvoke.return_value = AIMessage(content=result_obj.model_dump_json())

    # Correct paths
    agent_dir = plan_env["session_path"] / "agents" / "s1"
    inbox, outbox, todo = agent_dir/"inbox", agent_dir/"outbox", agent_dir/"todo"
    state = create_initial_state("s1", plan_env["user_id"], plan_env["session_id"], inbox, outbox, todo_path=todo, role=AgentRole.SUPERVISOR)
    
    # 2. Run Planning Node
    res = await sup.planner_node(state)
    
    # Verify task was added to TODO
    todo_mgr = TODOManager(todo.parent)
    tasks = todo_mgr.list_tasks()
    assert any(t.assigned_to == "new_helper" for t in tasks)

@pytest.mark.asyncio
async def test_supervisor_planning_tool_use(plan_env):
    """Verifies that supervisor chooses TOOL_USE for simple tasks."""
    sup = plan_env["supervisor"]
    
    # 1. Mock TOOL_USE strategy
    result_obj = PlanningResult(
        thought_process="Simple task, I'll do it myself.",
        action_sequence=[
            Action(strategy=ExecutionStrategy.TOOL_USE, name="ls", args={"path": "."})
        ]
    )
    # The initial call to self.llm.ainvoke in planner_node expects an object with a .content attribute.
    plan_env["mock_llm"].ainvoke.return_value = AIMessage(content=result_obj.model_dump_json())

    # Correct paths
    agent_dir = plan_env["session_path"] / "agents" / "s1"
    inbox, outbox, todo = agent_dir/"inbox", agent_dir/"outbox", agent_dir/"todo"
    state = create_initial_state("s1", plan_env["user_id"], plan_env["session_id"], inbox, outbox, todo_path=todo, role=AgentRole.SUPERVISOR)
    
    # 2. Run Planning Node
    res = await sup.planner_node(state)
    
    # Verify tool task was added to TODO
    todo_mgr = TODOManager(todo.parent)
    tasks = todo_mgr.list_tasks()
    assert any(t.type == TaskType.TOOL and t.payload["name"] == "ls" for t in tasks)
