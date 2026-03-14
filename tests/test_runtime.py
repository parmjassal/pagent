import pytest
from pathlib import Path
import shutil
import json
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.mailbox import FilesystemMailboxProvider

@pytest.fixture
def temp_workspace(tmp_path):
    """Creates a temporary .pagent root with global and user resources."""
    root = tmp_path / ".pagent"
    global_dir = root / "global"
    user_dir = root / "user_test_user"

    # Create dummy global resources
    (global_dir / "skills").mkdir(parents=True)
    (global_dir / "prompts").mkdir(parents=True)
    (global_dir / "skills" / "global_skill.py").write_text("global_skill_content")
    (global_dir / "prompts" / "base_prompt.txt").write_text("base_prompt_content")

    # Create dummy user resources (including an override)
    (user_dir / "skills").mkdir(parents=True)
    (user_dir / "prompts").mkdir(parents=True)
    (user_dir / "skills" / "user_skill.py").write_text("user_skill_content")
    (user_dir / "prompts" / "base_prompt.txt").write_text("user_override_content")

    return root

def test_workspace_hierarchy(temp_workspace):
    ctx = WorkspaceContext(root=temp_workspace)
    assert ctx.get_global_dir() == temp_workspace / "global"
    assert ctx.get_user_dir("user1") == temp_workspace / "user_user1"
    
    session_dir = ctx.get_session_dir("user1", "sess1")
    assert session_dir == temp_workspace / "user_user1" / "sess1"

def test_session_initialization_and_resource_copy(temp_workspace):
    workspace = WorkspaceContext(root=temp_workspace)
    resource_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, resource_mgr)

    user_id = "test_user"
    session_id = "session_abc"
    session_path = initializer.initialize(user_id, session_id)

    # Verify directory structure
    assert (session_path / "agents").exists()
    assert (session_path / "skills").exists()
    assert (session_path / "prompts").exists()

    # Verify resource inheritance (Copy, not Symlink)
    global_skill = session_path / "skills" / "global_skill.py"
    user_skill = session_path / "skills" / "user_skill.py"
    overridden_prompt = session_path / "prompts" / "base_prompt.txt"

    assert global_skill.exists()
    assert global_skill.read_text() == "global_skill_content"
    assert not global_skill.is_symlink()

    assert user_skill.exists()
    assert user_skill.read_text() == "user_skill_content"

    # Verify override logic (User should override Global)
    assert overridden_prompt.read_text() == "user_override_content"

def test_filesystem_mailbox_provider(temp_workspace):
    session_root = temp_workspace / "user_test" / "sess_test"
    provider = FilesystemMailboxProvider(session_root)
    
    agent_id = "agent_007"
    message = {"id": "msg1", "content": "hello"}
    
    # Send message
    provider.send(agent_id, message)
    
    inbox_path = session_root / "agents" / agent_id / "inbox"
    assert (inbox_path / "msg_msg1.json").exists()
    
    # Read message
    received = provider.read_inbox(agent_id)
    assert received["content"] == "hello"
