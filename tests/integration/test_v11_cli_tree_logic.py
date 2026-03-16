import pytest
from pathlib import Path
from rich.console import Console
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.cli import build_dynamic_tree

@pytest.fixture
def cli_test_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "ui_tester", "sess_ui_v11"
    session_path = initializer.initialize(user_id, session_id)

    # 1. Create a real hierarchy on disk
    # super
    # └── worker_1
    #     └── sub_worker_2
    
    workspace.ensure_agent_structure(user_id, session_id, "super")
    workspace.ensure_agent_structure(user_id, session_id, "worker_1")
    workspace.ensure_agent_structure(user_id, session_id, "sub_worker_2")

    # Create todo files for agents to define tasks and hierarchy
    super_todo_path = session_path / "agents" / "super" / "todo"
    worker_1_todo_path = session_path / "agents" / "worker_1" / "todo"
    sub_worker_2_todo_path = session_path / "agents" / "sub_worker_2" / "todo"

    super_todo_path.mkdir(parents=True, exist_ok=True)
    worker_1_todo_path.mkdir(parents=True, exist_ok=True)
    sub_worker_2_todo_path.mkdir(parents=True, exist_ok=True)

    # super agent tasks
    (super_todo_path / "task_1.json").write_text(
        '{"description": "Initial planning phase", "status": "completed", "assigned_to": null}'
    )
    (super_todo_path / "task_2.json").write_text(
        '{"description": "Delegate to worker 1", "status": "in_progress", "assigned_to": "worker_1"}'
    )

    # worker_1 agent tasks
    (worker_1_todo_path / "task_a.json").write_text(
        '{"description": "Execute sub-plan A", "status": "completed", "assigned_to": null}'
    )
    (worker_1_todo_path / "task_b.json").write_text(
        '{"description": "Coordinate with sub_worker 2", "status": "in_progress", "assigned_to": "sub_worker_2"}'
    )
    (worker_1_todo_path / "task_c.json").write_text(
        '{"description": "Final review", "status": "pending", "assigned_to": null}'
    )

    # sub_worker_2 agent tasks
    (sub_worker_2_todo_path / "task_x.json").write_text(
        '{"description": "Perform task X", "status": "in_progress", "assigned_to": null}'
    )

    return {
        "user_id": user_id, 
        "session_id": session_id, 
        "session_path": session_path,
        "workspace": workspace
    }

def test_v11_build_dynamic_tree_filesystem_sync(cli_test_env):
    """
    Validates that build_dynamic_tree correctly reflects the 
    real hierarchical filesystem structure.
    """
    env = cli_test_env
    
    # Call the real CLI tree builder
    tree = build_dynamic_tree(
        session_id=env["session_id"],
        user_id=env["user_id"],
        model_name="gpt-4o",
        task="Test Task",
        session_path=env["session_path"]
    )
    
    # Use rich Console to render the tree to text for assertion
    console = Console(width=100, color_system=None) # No colors for easier regex
    with console.capture() as capture:
        console.print(tree)
    
    output = capture.get()
    
    # 1. Assert Infrastructure is present and correctly formatted
    assert "🔌 Infrastructure" in output
    assert "👤 User: ui_tester" in output
    assert "🧠 Model: gpt-4o" in output
    assert "🎯 Initial Task: Test Task..." in output
    
    # 2. Assert Agent Hierarchy is present and correctly identified
    assert "🤖 Agent & Task Tree" in output
    assert "👑 Root Agent: super" in output
    
    # 3. Assert specific tasks and delegations
    # Super agent's tasks
    assert "📝 Task: Initial planning phase (completed)" in output
    assert "🤖 Agent: worker_1 - Goal: Delegate to worker 1 (in_progress)" in output
    
    # worker_1's tasks (delegated by super)
    assert "📝 Task: Execute sub-plan A (completed)" in output
    assert "🤖 Agent: sub_worker_2 - Goal: Coordinate with sub_worker 2 (in_progress)" in output
    assert "📝 Task: Final review (pending)" in output
    
    # sub_worker_2's tasks (delegated by worker_1)
    assert "📝 Task: Perform task X (in_progress)" in output

    # 4. Verify visual hierarchy (using careful string matching and order)
    # The output string has line breaks, so we can check nesting visually
    
    # super -> Delegate to worker 1 (worker_1)
    super_agent_line = output.find("👑 Root Agent: super")
    delegate_to_worker_1_line = output.find("🤖 Agent: worker_1 - Goal: Delegate to worker 1 (in_progress)")
    assert super_agent_line < delegate_to_worker_1_line
    
    # worker_1 -> Coordinate with sub_worker 2 (sub_worker_2)
    worker_1_agent_line = output.find("🤖 Agent: worker_1 - Goal: Delegate to worker 1 (in_progress)") # This is the parent node for worker_1
    coordinate_sub_worker_2_line = output.find("🤖 Agent: sub_worker_2 - Goal: Coordinate with sub_worker 2 (in_progress)")
    assert worker_1_agent_line < coordinate_sub_worker_2_line
    
    # The tasks for worker_1 should be indented under it
    execute_sub_plan_A_line = output.find("📝 Task: Execute sub-plan A (completed)")
    assert worker_1_agent_line < execute_sub_plan_A_line and execute_sub_plan_A_line < coordinate_sub_worker_2_line # crude check

    # The tasks for sub_worker_2 should be indented under it
    perform_task_X_line = output.find("📝 Task: Perform task X (in_progress)")
    assert coordinate_sub_worker_2_line < perform_task_X_line
