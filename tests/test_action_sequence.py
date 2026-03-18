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
from agent_platform.runtime.core.todo import TODOManager, TaskType, TaskStatus, ScopedTask
from langchain_core.messages import AIMessage

@pytest.fixture
def plan_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "action_tester", "sess_action"
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

@pytest.mark.asyncio
async def test_supervisor_action_sequence(plan_env):
    """Verifies that supervisor can handle a sequence of actions with per-action strategy."""
    sup = plan_env["supervisor"]
    
    # 1. Mock AUTHORIZE + DECOMPOSE sequence
    result_obj = PlanningResult(
        thought_process="Authorizing then decomposing.",
        action_sequence=[
            Action(strategy=ExecutionStrategy.AUTHORIZE, name="update_context", args={"fact_id": "auth", "content": "allowed"}),
            Action(strategy=ExecutionStrategy.DECOMPOSE, sub_tasks=[
                SubAgentTask(agent_id="sub1", role=AgentRole.WORKER, instructions="Do it")
            ])
        ]
    )
    plan_env["mock_llm"].ainvoke.return_value = AIMessage(content=result_obj.model_dump_json())

    # Correct paths
    agent_dir = plan_env["session_path"] / "agents" / "s1"
    inbox, outbox, todo = agent_dir/"inbox", agent_dir/"outbox", agent_dir/"todo"
    state = create_initial_state("s1", plan_env["user_id"], plan_env["session_id"], inbox, outbox, todo_path=todo, role=AgentRole.SUPERVISOR)
    
    # 2. Run Planning Node
    res = await sup.planner_node(state)
    
    # Verify tasks were added to TODO in sequence
    todo_mgr = TODOManager(todo.parent)
    tasks = todo_mgr.list_tasks()
    
    assert len(tasks) == 2
    assert tasks[0].payload["name"] == "update_context"
    assert tasks[1].type == TaskType.AGENT
    assert tasks[1].assigned_to == "sub1"

@pytest.mark.asyncio
async def test_supervisor_finish_in_sequence(plan_env):
    """Verifies that finish in sequence works."""
    sup = plan_env["supervisor"]
    
    result_obj = PlanningResult(
        thought_process="Tool then finish.",
        action_sequence=[
            Action(strategy=ExecutionStrategy.TOOL_USE, name="ls", args={"path": "."}),
            Action(strategy=ExecutionStrategy.FINISH, final_answer="Done.")
        ]
    )
    plan_env["mock_llm"].ainvoke.return_value = AIMessage(content=result_obj.model_dump_json())

    agent_dir = plan_env["session_path"] / "agents" / "s2"
    inbox, outbox, todo = agent_dir/"inbox", agent_dir/"outbox", agent_dir/"todo"
    state = create_initial_state("s2", plan_env["user_id"], plan_env["session_id"], inbox, outbox, todo_path=todo, role=AgentRole.SUPERVISOR)
    
    await sup.planner_node(state)
    
    todo_mgr = TODOManager(todo.parent)
    tasks = todo_mgr.list_tasks()
    assert len(tasks) == 2
    assert tasks[1].type == TaskType.FINISH
    assert tasks[1].payload["final_answer"] == "Done."

@pytest.mark.asyncio
async def test_failure_clears_todo(plan_env):
    """Verifies that a failure in one task clears the remaining tasks in the TODO."""
    sup = plan_env["supervisor"]
    
    agent_dir = plan_env["session_path"] / "agents" / "s3"
    inbox, outbox, todo = agent_dir/"inbox", agent_dir/"outbox", agent_dir/"todo"
    state = create_initial_state("s3", plan_env["user_id"], plan_env["session_id"], inbox, outbox, todo_path=todo, role=AgentRole.SUPERVISOR)
    
    todo_mgr = TODOManager(todo.parent)
    t1 = todo_mgr.add_task(ScopedTask(title="T1", description="D1", type=TaskType.TOOL))
    t2 = todo_mgr.add_task(ScopedTask(title="T2", description="D2", type=TaskType.TOOL))
    
    # Simulate T1 failure in collector_node
    state["metadata"]["next_task"] = {"task_id": t1, "type": TaskType.TOOL}
    state["metadata"]["task_error"] = "Something went wrong"
    
    await sup.collector_node(state)
    
    tasks = todo_mgr.list_tasks()
    t2_task = next(t for t in tasks if t.task_id == t2)
    assert t2_task.status == TaskStatus.FAILED
    assert "Aborted" in t2_task.result["output"]
