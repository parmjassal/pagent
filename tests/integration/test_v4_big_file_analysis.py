import pytest
from pathlib import Path
from unittest.mock import MagicMock
import json
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state
from agent_platform.runtime.storage.semantic_search import SemanticSearchEngine
from agent_platform.runtime.agents.search_agent import SemanticSearchAgent
from agent_platform.runtime.agents.fact_sheet_agent import FactSheetAgent
from agent_platform.runtime.storage.knowledge import FilesystemKnowledgeManager
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def big_file_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)
    
    data_dir = tmp_path / "logs"
    data_dir.mkdir()
    log_file = data_dir / "system.log"
    lines = [f"Line {i}: {'ERROR: Database Timeout' if i == 150 else 'Normal operation'}" for i in range(1, 251)]
    log_file.write_text("\n".join(lines))

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "analyst", "sess_v4_logs"
    session_path = initializer.initialize(user_id, session_id)

    km = FilesystemKnowledgeManager(session_path / "knowledge")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "FACT: Database timeout occurred at line 150 due to Connection pooling exhaustion."
    
    search_agent = SemanticSearchAgent(workspace)
    fact_agent = FactSheetAgent(km, llm=mock_llm)

    return {
        "user_id": user_id,
        "session_id": session_id,
        "session_path": session_path,
        "log_path": data_dir,
        "search_agent": search_agent,
        "fact_agent": fact_agent,
        "mock_llm": mock_llm
    }

def test_v4_big_file_analysis_flow(big_file_env):
    env = big_file_env
    search_agent = env["search_agent"]
    fact_agent = env["fact_agent"]
    
    state = create_initial_state(
        "analyst_01", env["user_id"], env["session_id"], 
        Path("/tmp"), Path("/tmp"), env["session_path"] / "knowledge"
    )

    state["metadata"]["target_folder"] = str(env["log_path"])
    index_res = search_agent.index_node(state)
    assert index_res["metadata"]["index_ready"] is True
    
    index_file = env["session_path"] / "semantic_index" / "index.json"
    with open(index_file, "r") as f:
        idx_data = json.load(f)
        assert len(idx_data["documents"]) >= 3

    state.update(index_res)
    state["metadata"]["search_query"] = "database timeout error"
    query_res = search_agent.query_node(state)
    
    best_chunk = query_res["search_results"][0]
    assert best_chunk["metadata"]["start_line"] <= 150 <= best_chunk["metadata"]["end_line"]

    state.update(query_res)
    state["metadata"]["current_chunk"] = best_chunk["metadata"]
    
    extract_res = fact_agent.extract_fact_node(state)
    assert "Extracted and persisted" in extract_res["messages"][0]["content"]
    
    fact_id = extract_res["metadata"]["last_fact_id"]
    fact_file = env["session_path"] / "knowledge" / f"{fact_id}.md"
    assert fact_file.exists()
    
    content = fact_file.read_text()
    assert "Connection pooling exhaustion" in content
