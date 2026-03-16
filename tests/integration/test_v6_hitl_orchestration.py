import pytest
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state
from agent_platform.runtime.agents.orchestrator import OrchestratorAgent
from agent_platform.runtime.core.hitl import InteractionManager, HITLRequest, HITLResponse, InteractionStatus
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def hitl_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "hitl_user", "sess_hitl_001"
    session_path = initializer.initialize(user_id, session_id)

    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    interaction_mgr = InteractionManager(session_path)

    return {
        "user_id": user_id, "session_id": session_id, "session_path": session_path,
        "factory": factory, "mailbox": mailbox, "interaction_mgr": interaction_mgr
    }

def test_v6_hitl_suspend_and_resume(hitl_env):
    env = hitl_env
    agent_id = "agent_with_hitl"
    
    # 1. Setup Persistence
    db_path = env["factory"].get_agent_db_path(env["user_id"], env["session_id"], agent_id)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    # 2. Setup Orchestrator with a Mocked LLM
    supervisor = OrchestratorAgent(env["factory"], env["mailbox"], MagicMock(), llm=MagicMock())
    graph = supervisor.build_graph(checkpointer=checkpointer)
    
    initial_state = create_initial_state(agent_id, env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
    config = {"configurable": {"thread_id": agent_id}}

    # --- PHASE 1: SUBMIT REQUEST & SUSPEND ---
    req = HITLRequest(agent_id=agent_id, context="Approval needed for destructive tool", data={"tool": "delete_all"})
    env["interaction_mgr"].submit_request(req)
    
    # Verify request is pending on disk
    pending = env["interaction_mgr"].list_pending()
    assert len(pending) == 1
    assert pending[0].request_id == req.request_id

    # In a real LangGraph, we would call 'interrupt()' inside a node.
    # Here, we update the state to include the interaction_id
    graph.update_state(config, {"active_interactions": [req.request_id]})

    # --- PHASE 2: HUMAN RESOLUTION ---
    response = HITLResponse(request_id=req.request_id, approved=True, feedback="Safe to proceed.")
    resolved_req = env["interaction_mgr"].resolve_request(response)
    assert resolved_req.status == InteractionStatus.APPROVED

    # --- PHASE 3: RESUME ---
    resumed_state = graph.get_state(config)
    assert req.request_id in resumed_state.values["active_interactions"]
    
    # Verify the decision is available to the agent now
    # The scheduler would ideally clear the active_interactions after handling the response
    graph.update_state(config, {"active_interactions": []}) # Simulation of cleanup
    final_state = graph.get_state(config)
    assert len(final_state.values["active_interactions"]) == 0

    conn.close()
