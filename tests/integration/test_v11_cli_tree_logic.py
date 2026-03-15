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
    workspace.ensure_agent_structure(user_id, session_id, "super/worker_1")
    workspace.ensure_agent_structure(user_id, session_id, "super/worker_1/sub_worker_2")
    
    # 2. Add some "Responded" state to worker_1
    outbox_path = session_path / "agents" / "super" / "worker_1" / "outbox"
    (outbox_path / "result.json").write_text('{"status": "done"}')

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
    
    # 1. Assert Infrastructure is present
    assert "Infrastructure" in output
    assert "ui_tester" in output
    assert "gpt-4o" in output
    
    # 2. Assert Agent Hierarchy is present and correctly identified
    assert "Agent Tree" in output
    assert "Agent: super" in output
    assert "Agent: worker_1" in output
    assert "Agent: sub_worker_2" in output
    
    # 3. Assert status logic works
    # worker_1 has a file in its outbox, so it should be "Responded"
    assert "worker_1 (Responded)" in output
    # others should be "Active"
    assert "super (Active)" in output
    assert "sub_worker_2 (Active)" in output

    # 4. Verify visual hierarchy (crude string check)
    # sub_worker_2 should appear AFTER worker_1
    worker_idx = output.find("worker_1")
    sub_worker_idx = output.find("sub_worker_2")
    assert worker_idx < sub_worker_idx
