import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agent_platform.runtime.core.workspace import WorkspaceContext
from agent_platform.runtime.core.resource_manager import SimpleCopyResourceManager, SessionInitializer
from agent_platform.runtime.core.agent_factory import AgentFactory
from agent_platform.runtime.orch.state import create_initial_state, AgentRole
from agent_platform.runtime.agents.supervisor import SupervisorAgent
from agent_platform.runtime.agents.worker import WorkerAgent
from agent_platform.runtime.agents.generator import SystemGeneratorAgent
from agent_platform.runtime.orch.unit_compiler import UnitCompiler
from agent_platform.runtime.orch.result_hook import OffloadingResultHook
from agent_platform.runtime.orch.models import PlanningResult, WorkerResult, ExecutionStrategy, SubAgentTask, ToolCall
from agent_platform.runtime.core.mailbox import Mailbox, FilesystemMailboxProvider
from agent_platform.runtime.core.dispatcher import ToolDispatcher, ToolRegistry
from agent_platform.runtime.storage.search_tool import SearchTools
from agent_platform.runtime.core.tools.filesystem import FilesystemTools
from agent_platform.runtime.core.context_store import FilesystemContextStore
from agent_platform.runtime.storage.context_tool import ContextTools
from agent_platform.runtime.core.guardrails import GuardrailManager, PolicyGenerator
from agent_platform.runtime.orch.tool_node import AgentToolNode

@pytest.fixture
def v10_env(tmp_path):
    root = tmp_path / ".pagent"
    workspace = WorkspaceContext(root=root)
    (workspace.get_global_dir() / "prompts").mkdir(parents=True)

    res_mgr = SimpleCopyResourceManager()
    initializer = SessionInitializer(workspace, res_mgr)
    user_id, session_id = "cloud_architect", "sess_v10"
    session_path = initializer.initialize(user_id, session_id)

    # 1. Setup mock repository INSIDE 'super' workspace
    agent_dir = workspace.ensure_agent_structure(user_id, session_id, "super")
    repo_path = agent_dir / "cloud_repo"
    repo_path.mkdir()
    
    # Java Code
    (repo_path / "Processor.java").write_text("""
        public class Processor {
            private DynamoDbClient ddb;
            private SqsClient sqs;
            public void handle() { ddb.putItem(...); sqs.sendMessage(...); }
        }
    """)
    # Terraform Code
    (repo_path / "lambda.tf").write_text("""
        resource "aws_lambda_function" "order_proc" {
            function_name = "OrderProcessor"
            reserved_concurrent_executions = 5
        }
    """)

    factory = AgentFactory(workspace)
    mailbox = Mailbox(FilesystemMailboxProvider(session_path))
    result_hook = OffloadingResultHook(session_path / "knowledge")
    
    # 2. Tooling
    registry = ToolRegistry(session_path)
    fs_tools = FilesystemTools(session_path)
    search_tools = SearchTools(session_path)
    context_store = FilesystemContextStore(session_path)
    context_tools = ContextTools(context_store)
    
    registry.register_native("read_file", fs_tools.read_file)
    registry.register_native("build_index", search_tools.build_index)
    registry.register_native("semantic_search", search_tools.semantic_search)
    registry.register_native("update_context", context_tools.update_context)

    mock_policy = MagicMock(spec=PolicyGenerator)
    mock_policy.generate.return_value = (True, "Allowed")
    dispatcher = ToolDispatcher(registry, MagicMock(), GuardrailManager(policy_generator=mock_policy))
    
    return {
        "user_id": user_id, "session_id": session_id, "session_path": session_path,
        "repo_path": repo_path, "factory": factory, "mailbox": mailbox,
        "result_hook": result_hook, "dispatcher": dispatcher, "workspace": workspace
    }

@pytest.mark.asyncio
async def test_v10_triplet_extraction_workflow(v10_env):
    env = v10_env
    
    # --- MOCK LLM TURNS ---
    
    # Supervisor logic
    mock_sup_llm = AsyncMock()
    mock_sup_llm.ainvoke.side_effect = [
        # Turn 1: Build Index
        PlanningResult(
            thought_process="Indexing codebase.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="build_index", args={"path": str(env["repo_path"])})
        ),
        # Turn 2: Decompose task to expert
        PlanningResult(
            thought_process="Spawning analyst for triplets.",
            strategy=ExecutionStrategy.DECOMPOSE,
            sub_tasks=[SubAgentTask(agent_id="analyst_01", role=AgentRole.WORKER, instructions="Extract cloud triplets.")]
        ),
        # Turn 3: Received result -> Store in context
        PlanningResult(
            thought_process="Received triplets. Storing in global context.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="update_context", args={
                "fact_id": "arch_triplets", 
                "content": "(OrderProcessor, uses, DynamoDB), (OrderProcessor, publishes, SQS), (OrderProcessor, concurrency, 5)"
            })
        ),
        # Turn 4: Done
        PlanningResult(thought_process="Audit done.", strategy=ExecutionStrategy.FINISH)
    ]

    # Worker logic
    mock_work_llm = AsyncMock()
    mock_work_llm.ainvoke.side_effect = [
        # Worker Turn 1: Search Java
        WorkerResult(
            thought_process="Searching Java logic.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="semantic_search", args={"query": "DynamoDB SQS"})
        ),
        # Worker Turn 2: Search Infra
        WorkerResult(
            thought_process="Searching Terraform logic.",
            strategy=ExecutionStrategy.TOOL_USE,
            tool_call=ToolCall(name="semantic_search", args={"query": "lambda concurrency"})
        ),
        # Worker Turn 3: Final Answer
        WorkerResult(
            thought_process="Extraction complete.",
            strategy=ExecutionStrategy.FINISH,
            final_answer="Extracted triplets for OrderProcessor."
        )
    ]

    # 3. Setup UnitCompiler with injected mocks
    mock_generator = MagicMock(spec=SystemGeneratorAgent)
    mock_generator.generate_node = AsyncMock()
    mock_generator.generate_node.return_value = {"messages": [], "generated_output": "MOCK PROMPT"}

    class V10UnitCompiler(UnitCompiler):
        def compile_unit(self, role, checkpointer=None):
            tool_node = AgentToolNode(env["dispatcher"])
            if role == AgentRole.SUPERVISOR:
                return SupervisorAgent(env["factory"], env["mailbox"], mock_generator, llm=mock_sup_llm, unit_compiler=self).build_graph(checkpointer=checkpointer, tool_node=tool_node)
            else:
                return WorkerAgent(tool_node, llm=mock_work_llm).build_graph()

    compiler = V10UnitCompiler(env["factory"], env["mailbox"], mock_generator, env["dispatcher"], env["result_hook"])
    
    agent_dir = env["workspace"].get_agent_dir(env["user_id"], env["session_id"], "super")
    initial_state = create_initial_state(
        "super", env["user_id"], env["session_id"],
        inbox_path=agent_dir / "inbox",
        outbox_path=agent_dir / "outbox",
        todo_path=agent_dir / "todo",
        role=AgentRole.SUPERVISOR
    )
    
    graph = compiler.compile_unit(AgentRole.SUPERVISOR)
    final_state = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": "v10_thread"}})

    # --- VERIFICATION ---
    
    # 1. Verify triplets were extracted and stored in hierarchical context
    fact_path = env["session_path"] / "agents" / "super" / "global_context" / "arch_triplets.md"
    assert fact_path.exists()
    content = fact_path.read_text()
    assert "(OrderProcessor, uses, DynamoDB)" in content
    assert "(OrderProcessor, concurrency, 5)" in content

    # 2. Verify worker result was bubbled up
    assert any("analyst_01 returned" in m["content"] for m in final_state["messages"] if m["role"] == "system")
    
    # 3. Verify semantic index exists
    assert (env["session_path"] / "semantic_index").exists()
