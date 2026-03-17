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
    
    # 1. Create a real hierarchy on disk to test orphans and nesting
    # super
    # └── (pending real_child)
    # orphan_worker
    
    # Create agents with a nested structure
    workspace.ensure_agent_structure(user_id, session_id, "supervisor")
    workspace.ensure_agent_structure(user_id, session_id, "orphan_worker")
    
    # 2. Create todo files to define tasks
    super_todo_path = session_path / "agents" / "super" / "todo"
    orphan_todo_path = session_path / "agents" / "orphan_worker" / "todo"

    super_todo_path.mkdir(parents=True, exist_ok=True)
    orphan_todo_path.mkdir(parents=True, exist_ok=True)

    # super agent tasks
    (super_todo_path / "task_1.json").write_text(
        '{"description": "Delegate to a real child", "status": "in_progress", "assigned_to": "supervisor/real_child"}'
    )
    long_path = "/a/very/long/path/that/is/definitely/going/to/be/longer/than/forty/characters/file.txt"
    (super_todo_path / "task_2.json").write_text(
        '{"description": "Invoke a tool with long value", "status": "completed", "type": "tool", "payload": {"name": "my_tool", "args": {"long_path": "' + long_path + '"}}}'
    )

    # orphan_worker tasks
    (orphan_todo_path / "task_x.json").write_text(
        '{"description": "Perform orphan task", "status": "pending", "assigned_to": null}'
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

    # 3. Assert that 'supervisor' is the only root agent
    assert "👑 Root Agent: supervisor" in output
    assert "👑 Root Agent: orphan_worker" not in output, "Orphan worker should not be a root agent"
    
    # 4. Assert tasks for 'supervisor'
    assert "🤖 Agent: real_child - Goal: Delegate to a real child (in_progress) (pending creation)" in output
    # Check for the correctly trimmed long path value (last 37 chars)
    assert "📝 Task: Invoke a tool with long value (long_path: ...longer/than/forty/characters/file.txt) (completed)" in output
    
    # 5. Assert that the 'Orphan Agents' section exists and contains the orphan
    assert "⚠️ Orphan Agents" in output
    assert "🤷 orphan_worker" in output
    assert "📝 Task: Perform orphan task (pending)" in output
    
    # 6. Verify visual hierarchy
    orphan_section_idx = output.find("⚠️ Orphan Agents")
    orphan_worker_idx = output.find("🤷 orphan_worker")
    assert orphan_section_idx > 0 and orphan_worker_idx > orphan_section_idx, "Orphan worker should be listed under the Orphan Agents section"
