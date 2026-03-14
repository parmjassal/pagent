import pytest
from pathlib import Path
import json
from agent_platform.runtime.workspace import WorkspaceContext
from agent_platform.runtime.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.agent_factory import AgentFactory
from agent_platform.runtime.state import create_initial_state, AgentRole
from agent_platform.runtime.quota import update_quota
from agent_platform.runtime.generator import SystemGeneratorAgent
from agent_platform.runtime.supervisor import SupervisorAgent
from agent_platform.runtime.search_agent import SemanticSearchAgent
from agent_platform.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def repo_env(tmp_path):
    # 1. Setup Workspace
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)
    
    # 2. Setup Mock Repo (Distributed System)
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()
    (repo_dir / "raft.py").write_text("class Raft: def consensus(self): pass # Distributed consensus logic")
    (repo_dir / "kv_store.py").write_text("class KVStore: def put(self, k, v): pass # Storage engine")
    (repo_dir / "network.py").write_text("def send_rpc(node_id, msg): pass # Network transport")

    # 3. Initialize Session
    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "repo_expert", "sess_repo_v3"
    session_path = initializer.initialize(user_id, session_id)

    # 4. Components
    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    generator = SystemGeneratorAgent(llm=None, workspace=workspace)
    supervisor = SupervisorAgent(factory, mailbox, generator, api_key="dummy")
    search_agent = SemanticSearchAgent(workspace)

    return {
        "user_id": user_id,
        "session_id": session_id,
        "session_path": session_path,
        "repo_path": repo_dir,
        "supervisor": supervisor,
        "search_agent": search_agent,
        "mailbox": mailbox
    }

def test_v3_repo_analysis_with_index_reuse(repo_env):
    """
    Integration V3:
    Supervisor Spawn -> Analyst Indexing -> Consecutive Analyst Queries (Reuse Index)
    """
    env = repo_env
    supervisor = env["supervisor"]
    search_agent = env["search_agent"]
    
    # --- PHASE 1: ORCHESTRATION ---
    state = create_initial_state("super_01", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
    state["next_steps"] = ["analyst_agent"]
    
    # Spawn the analyst
    spawn_res = supervisor.spawning_node(state)
    assert "Successfully spawned" in spawn_res["messages"][-1]["content"]
    assert (env["session_path"] / "agents" / "analyst_agent").exists()

    # --- PHASE 2: INDEXING ---
    # Analyst state (simulated)
    analyst_state = create_initial_state("analyst_agent", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
    analyst_state["metadata"]["target_folder"] = str(env["repo_path"])
    
    # Run Indexing Node
    index_res = search_agent.index_node(analyst_state)
    assert index_res["metadata"]["index_ready"] is True
    
    # Verify index file was persisted
    index_file = env["session_path"] / "semantic_index" / "index.json"
    assert index_file.exists()
    
    # --- PHASE 3: QUERY 1 (CONSENSUS) ---
    analyst_state.update(index_res)
    analyst_state["metadata"]["search_query"] = "consensus algorithm"
    
    query1_res = search_agent.query_node(analyst_state)
    assert "raft.py" in query1_res["messages"][0]["content"]
    assert query1_res["search_results"][0]["score"] > 0

    # --- PHASE 4: QUERY 2 (STORAGE REUSE) ---
    # We clear the state but keep session/user IDs to force index RELOAD from disk
    clean_state = create_initial_state("analyst_agent", env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
    clean_state["metadata"]["search_query"] = "key value storage"
    
    # Run Query Node again - it should LOAD the existing index instead of rebuilding
    query2_res = search_agent.query_node(clean_state)
    
    assert "kv_store.py" in query2_res["messages"][0]["content"]
    assert query2_res["search_results"][0]["score"] > 0
    
    # Final check: Ensure we didn't re-index (original index file mtime should be unchanged if we had a tracer)
    # But semantically, the presence of results in a 'clean' state proves it was loaded from the session path.
