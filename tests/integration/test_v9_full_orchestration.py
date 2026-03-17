import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.orch.quota import SessionQuota
from agent_platform.runtime.agents.generator import SystemGeneratorAgent
from agent_platform.runtime.agents.orchestrator import OrchestratorAgent
from agent_platform.runtime.orch.unit_compiler import UnitCompiler
from agent_platform.runtime.orch.result_hook import OffloadingResultHook
from agent_platform.runtime.orch.models import PlanningResult, ExecutionStrategy, SubAgentTask, ToolCall
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider
from agent_platform.runtime.core.guardrails import GuardrailManager, PolicyGenerator
from agent_platform.runtime.core.dispatcher import ToolDispatcher, ToolRegistry
from agent_platform.runtime.storage.search_tool import SearchTools
from agent_platform.runtime.core.tools.filesystem import FilesystemTools
from agent_platform.runtime.core.context_store import FilesystemContextStore
from agent_platform.runtime.storage.context_tool import ContextTools
from agent_platform.runtime.orch.tool_node import AgentToolNode

@pytest.fixture
def v9_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "lego_architect", "sess_v9"
    session_path = initializer.initialize(user_id, session_id)

    # Setup mock repository
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
    
    # 1. Mock LLM for Orchestrator (Supervisor role)
    mock_sup_llm = AsyncMock()
    mock_sup_llm.ainvoke.side_effect = [
        # Turn 1: Build Index (TOOL_USE)
        AIMessage(content=PlanningResult(
            thought_process="I need to index the repo first.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="build_index", args={"path": str(env["repo_path"])})
        ).model_dump_json()),
        # Turn 2: Spawn Worker (DECOMPOSE)
        AIMessage(content=PlanningResult(
            thought_process="Index built. Spawning worker to find logging patterns.",
            strategy=ExecutionStrategy.DECOMPOSE,
            sub_tasks=[SubAgentTask(agent_id="analyst", role=AgentRole.WORKER, instructions="Find logging standards.")]
        ).model_dump_json()),
        # Turn 3: Received worker result -> Update Context (TOOL_USE)
        AIMessage(content=PlanningResult(
            thought_process="Worker found 'standard' logger. Promoting to global context.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="update_context", args={"fact_id": "logging_std", "content": "The project uses 'standard' logger."})
        ).model_dump_json()),
        # Turn 4: Final verification and finish (FINISH)
        AIMessage(content=PlanningResult(
            thought_process="Audit complete.",
            strategy=ExecutionStrategy.FINISH
        ).model_dump_json())
    ]

    # 2. Mock LLM for Orchestrator (Worker role)
    mock_work_llm = AsyncMock()
    mock_work_llm.ainvoke.side_effect = [
        # Turn 1: Search repo (TOOL_USE)
        AIMessage(content=PlanningResult(
            thought_process="Searching for logging in repo.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="semantic_search", args={"query": "logging setup"})
        ).model_dump_json()),
        # Turn 2: Read file found (TOOL_USE)
        AIMessage(content=PlanningResult(
            thought_process="Found server.py. Reading it.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="read_file", args={"file_path": str(env["repo_path"] / "server.py")})
        ).model_dump_json()),
        # Turn 3: Finish (FINISH)
        AIMessage(content=PlanningResult(
            thought_process="Analyzed. It uses 'standard' logger.",
            strategy=ExecutionStrategy.FINISH
        ).model_dump_json())
    ]

    # 3. Setup Mock Generator
    mock_generator = MagicMock(spec=SystemGeneratorAgent)
    mock_generator.generate_node = AsyncMock(return_value={"generated_output": "MOCK PROMPT"})

    # 4. Use UnitCompiler to build the graph
    class MockUnitCompiler(UnitCompiler):
        def compile_unit(self, role, checkpointer=None):
            # In v3.0, both use OrchestratorAgent, just different LLM mocks for this test
            current_llm = mock_sup_llm if role == AgentRole.SUPERVISOR else mock_work_llm
            agent = OrchestratorAgent(
                env["factory"], env["mailbox"], mock_generator,
                llm=current_llm,
                unit_compiler=self, # For recursion
                result_hook=env["result_hook"],
                tool_manifest="MOCK TOOLS"
            )
            tool_node = AgentToolNode(env["dispatcher"])
            return agent.build_graph(checkpointer=checkpointer, tool_node=tool_node)

    compiler = MockUnitCompiler(env["factory"], env["mailbox"], mock_generator, env["dispatcher"], env["result_hook"])
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
    
    # B. Check if context was updated
    # In ContextTools.update_context, it writes to 'global_context' folder in agent dir
    fact_path = env["session_path"] / "agents" / "super" / "global_context" / "logging_std.md"
    assert fact_path.exists()
    assert "uses 'standard' logger" in fact_path.read_text()

    # C. Check if worker result bubbled up (in history)
    # Orchestrator's collector_node adds a message like: "[System] Task {task_id} finished: ..."
    # final_state["messages"] contains dicts and AIMessages
    def get_content(m):
        if isinstance(m, dict):
            return m.get("content", "")
        return getattr(m, "content", "")

    assert any("finished" in get_content(m) for m in final_state["messages"])
    assert final_state.get("final_result") is not None
    assert final_state["quota"].agent_count == 1 
