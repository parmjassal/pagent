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
    
    # 1. Create a real hierarchy on disk
    # super
    # └── worker_1
    # independent_agent
    
    # Create agents with a nested structure
    workspace.ensure_agent_structure(user_id, session_id, "super")
    workspace.ensure_agent_structure(user_id, session_id, "super/worker_1")
    workspace.ensure_agent_structure(user_id, session_id, "independent_agent")
    
    # 2. Create todo files to define tasks, hierarchy, and pending creation state
    super_todo_path = session_path / "agents" / "super" / "todo"
    worker_1_todo_path = session_path / "agents" / "super" / "worker_1" / "todo"
    independent_todo_path = session_path / "agents" / "independent_agent" / "todo"

    # super agent tasks
    (super_todo_path / "task_1.json").write_text(
        '{"description": "Delegate to worker 1", "status": "in_progress", "assigned_to": "super/worker_1"}'
    )
    (super_todo_path / "task_2.json").write_text(
        '{"description": "Delegate to a phantom agent", "status": "in_progress", "assigned_to": "phantom_agent"}'
    )

    # worker_1 agent tasks (nested)
    (worker_1_todo_path / "task_a.json").write_text(
        '{"description": "Execute nested task", "status": "completed", "assigned_to": null}'
    )

    # independent_agent tasks
    (independent_todo_path / "task_x.json").write_text(
        '{"description": "Perform independent task", "status": "pending", "assigned_to": null}'
    )
    (independent_todo_path / "task_y.json").write_text(
        '{"description": "Invoke a specific tool", "status": "completed", "type": "tool", "payload": {"name": "my_tool", "args": {"input_file": "/path/to/data.txt", "mode": "read"}}, "assigned_to": null}'
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
    console = Console(width=150, color_system=None) # No colors for easier regex
    with console.capture() as capture:
        console.print(tree)
    
    output = capture.get()
    
    # 1. Assert Infrastructure is present
    assert "🔌 Infrastructure" in output
    assert "👤 User: ui_tester" in output
    
    # 2. Assert Agent Hierarchy root is present
    assert "🤖 Agent & Task Tree" in output

    # 3. Assert the two root agents are found and displayed
    # Note: The order is sorted, so 'independent_agent' comes before 'super'
    assert "👑 Root Agent: independent_agent" in output
    assert "👑 Root Agent: super" in output
    
    # 4. Assert tasks for 'independent_agent'
    assert "📝 Task: Perform independent task (pending)" in output
    assert "📝 Task: Invoke a specific tool (input_file: /path/to/data.txt) (completed)" in output
    
    # 5. Assert tasks and delegations for 'super'
    # Assert for the simple name 'worker_1' instead of the full path 'super/worker_1'
    assert "🤖 Agent: worker_1 - Goal: Delegate to worker 1 (in_progress)" in output
    assert "🤖 Agent: phantom_agent - Goal: Delegate to a phantom agent (in_progress) (pending creation)" in output

    # 6. Assert tasks for nested agent 'super/worker_1'
    assert "📝 Task: Execute nested task (completed)" in output

    # 7. Verify visual hierarchy (using careful string matching and order)
    super_root_idx = output.find("👑 Root Agent: super")
    # Find the delegation by its simple name
    worker_1_delegation_idx = output.find("🤖 Agent: worker_1")
    nested_task_idx = output.find("📝 Task: Execute nested task (completed)")
    phantom_delegation_idx = output.find("🤖 Agent: phantom_agent")

    assert super_root_idx < worker_1_delegation_idx
    assert worker_1_delegation_idx < nested_task_idx
    assert super_root_idx < phantom_delegation_idx
