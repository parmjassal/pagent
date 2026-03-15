import pytest
import aiosqlite
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state
from agent_platform.runtime.agents.generator import SystemGeneratorAgent
from agent_platform.runtime.agents.supervisor import SupervisorAgent
from agent_platform.runtime.orch.models import DecompositionResult, SubAgentTask
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider

@pytest.fixture
def persistence_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "resumable_user", "sess_persist_001"
    session_path = initializer.initialize(user_id, session_id)

    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = DecompositionResult(
        thought_process="Persistence Step 1",
        sub_tasks=[SubAgentTask(agent_id="worker_1", role="worker", instructions="Task")]
    )

    mock_gen_llm = AsyncMock()
    mock_gen_llm.ainvoke.return_value.content = "SYSTEM PROMPT"

    generator = SystemGeneratorAgent(llm=mock_gen_llm, workspace=workspace)
    
    return {
        "user_id": user_id, "session_id": session_id, "session_path": session_path,
        "factory": factory, "mailbox": mailbox, "generator": generator, 
        "mock_llm": mock_llm, "workspace": workspace
    }

@pytest.mark.asyncio
async def test_v5_graph_persistence_and_resume(persistence_env):
    env = persistence_env
    agent_id = "supervisor_01"
    
    db_path = env["factory"].get_agent_db_path(env["user_id"], env["session_id"], agent_id)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # USE ASYNC SQLITE
    async with aiosqlite.connect(db_path) as conn:
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
        
        supervisor = SupervisorAgent(env["factory"], env["mailbox"], env["generator"], llm=env["mock_llm"])
        graph = supervisor.build_graph(checkpointer=checkpointer)
        
        initial_state = create_initial_state(agent_id, env["user_id"], env["session_id"], Path("/tmp"), Path("/tmp"))
        config = {"configurable": {"thread_id": agent_id}}
        
        await graph.ainvoke(initial_state, config=config)
        
        # Verify resumed state
        resumed_state = await graph.aget_state(config)
        assert resumed_state.values["agent_id"] == agent_id
        assert resumed_state.values["quota"].agent_count == 1
