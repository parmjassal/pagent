import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.orch.quota import SessionQuota
from agent_platform.runtime.agents.generator import SystemGeneratorAgent
from agent_platform.runtime.agents.supervisor import SupervisorAgent
from agent_platform.runtime.agents.worker import WorkerAgent
from agent_platform.runtime.orch.unit_compiler import UnitCompiler
from agent_platform.runtime.orch.result_hook import OffloadingResultHook
from agent_platform.runtime.orch.models import PlanningResult, WorkerResult, ExecutionStrategy, SubAgentTask, ToolCall
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider
from agent_platform.runtime.core.guardrails import GuardrailManager, PolicyGenerator
from agent_platform.runtime.core.dispatcher import ToolDispatcher, ToolRegistry
from agent_platform.runtime.storage.search_tool import SearchTools
from agent_platform.runtime.core.tools.filesystem import FilesystemTools
from agent_platform.runtime.core.context_store import FilesystemContextStore
from agent_platform.runtime.storage.context_tool import ContextTools

@pytest.fixture
def v9_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "lego_architect", "sess_v9"
    session_path = initializer.initialize(user_id, session_id)

    # Setup mock repository INSIDE the agent workspace to pass boundary checks
    agent_dir = workspace.ensure_agent_structure(user_id, session_id, "super")
    repo_path = agent_dir / "mock_repo"
    repo_path.mkdir()
    (repo_path / "server.py").write_text("import logging\nlogger = logging.getLogger('standard')\nlogger.info('Started')")

    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    result_hook = OffloadingResultHook(session_path / "knowledge")
    
    # Tools Registration
    registry = ToolRegistry(session_path)
    fs_tools = FilesystemTools(session_path)
    search_tools = SearchTools(session_path)
    context_store = FilesystemContextStore(session_path)
    context_tools = ContextTools(context_store)
    
    registry.register_native("ls", fs_tools.ls)
    registry.register_native("read_file", fs_tools.read_file)
    registry.register_native("build_index", search_tools.build_index)
    registry.register_native("semantic_search", search_tools.semantic_search)
    registry.register_native("update_context", context_tools.update_context)
    registry.register_native("read_context", context_tools.read_context)
    registry.register_native("list_context", context_tools.list_context)

    # Guardrails setup
    mock_policy_gen = MagicMock(spec=PolicyGenerator)
    mock_policy_gen.generate.return_value = (True, "Allowed")
    guardrails = GuardrailManager(policy_generator=mock_policy_gen)

    dispatcher = ToolDispatcher(registry, MagicMock(), guardrails)
    
    return {
        "user_id": user_id, "session_id": session_id, "session_path": session_path,
        "repo_path": repo_path, "factory": factory, "mailbox": mailbox,
        "result_hook": result_hook, "dispatcher": dispatcher, "workspace": workspace
    }

@pytest.mark.asyncio
async def test_v9_full_architectural_audit_flow(v9_env):
    env = v9_env
    
    # 1. Mock LLM for Supervisor
    mock_sup_llm = AsyncMock()
    mock_sup_llm.ainvoke.side_effect = [
        # Turn 1: Build Index
        PlanningResult(
            thought_process="I need to index the repo first.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="build_index", args={"path": str(env["repo_path"])})
        ),
        # Turn 2: Spawn Worker
        PlanningResult(
            thought_process="Index built. Spawning worker to find logging patterns.",
            strategy=ExecutionStrategy.DECOMPOSE,
            sub_tasks=[SubAgentTask(agent_id="analyst", role=AgentRole.WORKER, instructions="Find logging standards.")]
        ),
        # Turn 3: Received worker result -> Update Context
        PlanningResult(
            thought_process="Worker found 'standard' logger. Promoting to global context.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="update_context", args={"fact_id": "logging_std", "content": "The project uses 'standard' logger."})
        ),
        # Turn 4: Final verification and finish
        PlanningResult(
            thought_process="Audit complete.",
            strategy=ExecutionStrategy.FINISH
        )
    ]

    # 2. Mock LLM for Worker
    mock_work_llm = AsyncMock()
    mock_work_llm.ainvoke.side_effect = [
        # Turn 1: Search repo
        WorkerResult(
            thought_process="Searching for logging in repo.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="semantic_search", args={"query": "logging setup"})
        ),
        # Turn 2: Read file found
        WorkerResult(
            thought_process="Found server.py. Reading it.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="read_file", args={"file_path": str(env["repo_path"] / "server.py")})
        ),
        # Turn 3: Finish with result
        WorkerResult(
            thought_process="Analyzed. It uses 'standard' logger.",
            strategy=ExecutionStrategy.FINISH,
            final_answer="The project uses 'standard' logger."
        )
    ]

    # 3. Setup UnitCompiler with injected mocks
    mock_generator = MagicMock(spec=SystemGeneratorAgent)
    mock_generator.generate_node = AsyncMock()
    mock_generator.generate_node.return_value = {"messages": [], "generated_output": "MOCK PROMPT"}

    class MockUnitCompiler(UnitCompiler):
        def compile_unit(self, role, checkpointer=None):
            if role == AgentRole.SUPERVISOR:
                return SupervisorAgent(env["factory"], env["mailbox"], mock_generator, llm=mock_sup_llm, unit_compiler=self).build_graph(checkpointer=checkpointer, tool_node=self.tool_node)
            else:
                return WorkerAgent(self.tool_node, llm=mock_work_llm).build_graph()

    compiler = MockUnitCompiler(env["factory"], env["mailbox"], mock_generator, env["dispatcher"], env["result_hook"])
    compiler.tool_node = MagicMock(side_effect=lambda state: env["dispatcher"].dispatch(state, **state["metadata"]["next_tool_call"]))
    # Wait, dispatcher.dispatch returns a dict, but AgentToolNode returns state updates.
    # Let's use the real AgentToolNode
    from agent_platform.runtime.orch.tool_node import AgentToolNode
    real_tool_node = AgentToolNode(env["dispatcher"])
    compiler.tool_node = real_tool_node

    # 4. Start Execution
    supervisor_graph = compiler.compile_unit(AgentRole.SUPERVISOR)
    
    agent_dir = env["workspace"].get_agent_dir(env["user_id"], env["session_id"], "super")
    initial_state = create_initial_state(
        "super", env["user_id"], env["session_id"], 
        inbox_path=agent_dir / "inbox", 
        outbox_path=agent_dir / "outbox",
        todo_path=agent_dir / "todo",
        role=AgentRole.SUPERVISOR
    )
    
    final_state = await supervisor_graph.ainvoke(initial_state, config={"configurable": {"thread_id": "v9_thread"}})

    # 5. Assertions
    # A. Check if index was built (Side effect of build_index)
    assert (env["session_path"] / "semantic_index").exists()
    
    # B. Check if global_context was updated (Side effect of update_context)
    fact_path = env["session_path"] / "agents" / "super" / "global_context" / "logging_std.md"
    assert fact_path.exists()
    assert "uses 'standard' logger" in fact_path.read_text()

    # C. Check if worker result bubbled up
    # The last system message should contain the result from Turn 3 merge
    assert any("analyst returned" in m["content"] for m in final_state["messages"] if m["role"] == "system")
    assert final_state["metadata"]["strategy"] == ExecutionStrategy.FINISH
    assert final_state["quota"].agent_count == 1 # Supervisor + Analyst (initially 0, spawns 1)
