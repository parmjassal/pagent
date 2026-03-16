import pytest
import json
from pathlib import Path
from agent_platform.runtime.storage.context_tool import ContextTools
from agent_platform.runtime.core.context_store import FilesystemContextStore
from agent_platform.runtime.core.tools.filesystem import FilesystemTools
from agent_platform.runtime.orch.result_hook import OffloadingResultHook
from agent_platform.runtime.orch.state import create_initial_state, AgentRole

@pytest.fixture
def session_env(tmp_path):
    session_path = tmp_path / "session"
    session_path.mkdir()
    knowledge_path = session_path / "knowledge"
    knowledge_path.mkdir()
    
    store = FilesystemContextStore(session_path)
    context_tools = ContextTools(store, knowledge_path=knowledge_path)
    fs_tools = FilesystemTools(session_path)
    
    return {
        "session_path": session_path,
        "knowledge_path": knowledge_path,
        "context_tools": context_tools,
        "fs_tools": fs_tools,
        "store": store
    }

def test_update_knowledge_prefixing(session_env):
    tools = session_env["context_tools"]
    state = create_initial_state("a1", "u1", "s1", Path("/tmp"), Path("/tmp"))
    
    res = tools.update_knowledge(state, "test_report", {"data": 123})
    assert "Successfully promoted" in res
    
    # Check file exists with prefix
    files = list(session_env["knowledge_path"].glob("*_test_report.json"))
    assert len(files) == 1
    assert len(files[0].name.split("_")[0]) == 8 # Hex prefix length
    
    content = json.loads(files[0].read_text())
    assert content["data"] == 123

def test_offloading_hook_prefixing(session_env):
    hook = OffloadingResultHook(session_env["knowledge_path"], threshold_bytes=10)
    
    res = hook.process_result("worker1", "This is a very long string that should be offloaded")
    assert res["type"] == "reference"
    assert "knowledge/offload_" in res["path"]
    
    # Check file
    files = list(session_env["knowledge_path"].glob("offload_*.json"))
    assert len(files) == 1
    assert "worker1" in files[0].name

def test_filesystem_write_restriction(session_env):
    fs = session_env["fs_tools"]
    
    # 1. Normal write OK
    res = fs.write_file("test.txt", "hello")
    assert res["success"] is True
    
    # 2. Direct write to knowledge/ blocked
    res = fs.write_file("knowledge/hack.json", "bad data")
    assert res["success"] is False
    assert "Direct writes to the 'knowledge/' directory are prohibited" in res["error"]
    
    # 3. Subfolder write blocked
    res = fs.write_file("knowledge/sub/hack.json", "bad data")
    assert res["success"] is False
    assert "Direct writes to the 'knowledge/' directory are prohibited" in res["error"]

def test_update_context_downward_flow(session_env):
    tools = session_env["context_tools"]
    # Supervisor updates context
    state_sup = create_initial_state("super", "u1", "s1", Path("/tmp"), Path("/tmp"), role=AgentRole.SUPERVISOR)
    tools.update_context(state_sup, "guideline", "Use JSON.")
    
    # Worker reads context (inherited)
    state_work = create_initial_state("super/worker1", "u1", "s1", Path("/tmp"), Path("/tmp"), role=AgentRole.WORKER)
    facts = tools.list_context(state_work)
    assert "guideline" in facts
    assert tools.read_context(state_work, "guideline") == "Use JSON."
