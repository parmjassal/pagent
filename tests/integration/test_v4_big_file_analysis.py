import pytest
from pathlib import Path
from unittest.mock import MagicMock
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.state import create_initial_state
from agent_platform.runtime.semantic_search import SemanticSearchEngine
from agent_platform.runtime.search_agent import SemanticSearchAgent
from agent_platform.runtime.fact_sheet_agent import FactSheetAgent
from agent_platform.runtime.knowledge import FilesystemKnowledgeManager
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def big_file_env(tmp_path):
    # 1. Setup Workspace
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)
    
    # 2. Setup Big File (250 lines of logs)
    data_dir = tmp_path / "logs"
    data_dir.mkdir()
    log_file = data_dir / "system.log"
    lines = [f"Line {i}: {'ERROR: Database Timeout' if i == 150 else 'Normal operation'}" for i in range(1, 251)]
    log_file.write_text("\n".join(lines))

    # 3. Initialize Session
    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "analyst", "sess_v4_logs"
    session_path = initializer.initialize(user_id, session_id)

    # 4. Components
    # We use a real knowledge manager but mock the LLM for the FactSheetAgent
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
    """
    Integration V4:
    Chunked Indexing -> Semantic Query -> Fact Extraction -> Knowledge Storage
    """
    env = big_file_env
    search_agent = env["search_agent"]
    fact_agent = env["fact_agent"]
    
    # Initial state
    state = create_initial_state(
        "analyst_01", env["user_id"], env["session_id"], 
        Path("/tmp"), Path("/tmp"), env["session_path"] / "knowledge"
    )

    # --- PHASE 1: CHUNKED INDEXING ---
    state["metadata"]["target_folder"] = str(env["log_path"])
    index_res = search_agent.index_node(state)
    assert index_res["metadata"]["index_ready"] is True
    
    # Verify index exists and has multiple chunks (250 lines / 100 line chunks = 3 chunks)
    index_file = env["session_path"] / "semantic_index" / "index.json"
    with open(index_file, "r") as f:
        idx_data = json.load(f)
        assert len(idx_data["documents"]) >= 3

    # --- PHASE 2: SEMANTIC QUERY ---
    state.update(index_res)
    state["metadata"]["search_query"] = "database timeout error"
    query_res = search_agent.query_node(state)
    
    # Ensure it found the relevant chunk (line 150)
    best_chunk = query_res["search_results"][0]
    assert best_chunk["metadata"]["start_line"] <= 150 <= best_chunk["metadata"]["end_line"]

    # --- PHASE 3: FACT EXTRACTION (Mocked LLM) ---
    state.update(query_res)
    # The agent expects 'current_chunk' in metadata to perform extraction
    state["metadata"]["current_chunk"] = best_chunk["metadata"]
    
    extract_res = fact_agent.extract_fact_node(state)
    assert "Extracted and persisted" in extract_res["messages"][0]["content"]
    
    # --- PHASE 4: VERIFY PERSISTENCE ---
    fact_id = extract_res["metadata"]["last_fact_id"]
    fact_file = env["session_path"] / "knowledge" / f"{fact_id}.md"
    assert fact_file.exists()
    
    content = fact_file.read_text()
    assert "Connection pooling exhaustion" in content
    assert "system.log#L101-200" in content # Should be in the 2nd chunk (101-200)

import json
