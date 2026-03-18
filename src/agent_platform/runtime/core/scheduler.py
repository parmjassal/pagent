import asyncio
import logging
import aiosqlite
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_openai import ChatOpenAI

from .workspace import WorkspaceContext
from .mailbox import Mailbox, FilesystemMailboxProvider
from .agent_factory import AgentFactory
from .dispatcher import ToolDispatcher, ToolRegistry
from .sandbox import ProcessSandboxRunner
from .guardrails import GuardrailManager, LLMPolicyGenerator
from .todo import TODOManager
from ..agents.orchestrator import OrchestratorAgent
from ..agents.generator import SystemGeneratorAgent, TaskType
from ..orch.state import create_initial_state, AgentRole
from ..orch.tool_node import AgentToolNode
from ..core.schema import ToolSource
from ..core.tools.filesystem import FilesystemTools
from ..core.context_store import FilesystemContextStore
from ..storage.context_tool import ContextTools
from ..storage.todo_tool import TODOTool
from ..orch.result_hook import OffloadingResultHook
from ..core.http_client import get_platform_async_http_client

logger = logging.getLogger(__name__)

class AutonomousScheduler:
    """
    Background scheduler that monitors mailboxes and triggers LangGraph agents.
    """
    def __init__(
        self, 
        workspace: WorkspaceContext, 
        user_id: str, 
        session_id: str,
        model_name: str = "gpt-4o",
        openai_base_url: Optional[str] = None
    ):
        self.workspace = workspace
        self.user_id = user_id
        self.session_id = session_id
        self.session_path = workspace.get_session_dir(user_id, session_id)
        self.mailbox = Mailbox(FilesystemMailboxProvider(self.session_path))
        self.factory = AgentFactory(workspace)
        self.model_config = {
            "model_name": model_name,
            "openai_base_url": openai_base_url
        }
        
        # Shared Context Store
        self.context_store = FilesystemContextStore(self.session_path)

        # Shared generator for the session
        self.generator = SystemGeneratorAgent(
            model_name=model_name,
            base_url=openai_base_url,
            workspace=workspace,
            context_store=self.context_store
        )

    async def run_forever(self):
        """Main loop of the scheduler."""
        logger.info(f"Scheduler started for session {self.session_id}")
        try:
            while True:
                try:
                    await self.tick()
                except Exception as e:
                    logger.error(f"Scheduler tick error: {e}", exc_info=True)
                await asyncio.sleep(2)
        finally:
            from .http_client import close_platform_async_http_client
            await close_platform_async_http_client()

    async def tick(self):
        """Recursively discovers and processes agents."""
        agents_root = self.session_path / "agents"
        if not agents_root.exists():
            return

        for root, dirs, files in os.walk(agents_root):
            root_path = Path(root)
            if (root_path / "inbox").exists():
                agent_id = str(root_path.relative_to(agents_root))
                await self._process_agent(agent_id)

    async def _process_agent(self, agent_id: str):
        inbox_msg = self.mailbox.receive(agent_id)
        if not inbox_msg:
            return

        logger.info(f"Scheduler: Triggering execution for agent '{agent_id}'")
        
        agent_dir = self.workspace.get_agent_dir(self.user_id, self.session_id, agent_id)
        db_path = agent_dir / "state.db"
        
        # Prepare paths for state
        inbox_path = agent_dir / "inbox"
        outbox_path = agent_dir / "outbox"
        todo_path = agent_dir / "todo"
        knowledge_path = self.session_path / "knowledge"

        async with aiosqlite.connect(db_path) as conn:
            checkpointer = AsyncSqliteSaver(conn)

            # Setup Tooling
            registry = ToolRegistry(self.session_path)

            # 1. Platform Core Tools (Priority 1: Reasoning & Context)
            todo_tool = TODOTool(self.session_path)
            registry.register_native("add_task", todo_tool.add_task, source=ToolSource.CORE)
            registry.register_native("list_tasks", todo_tool.list_tasks, source=ToolSource.CORE)
            registry.register_native("update_task_status", todo_tool.update_task_status, source=ToolSource.CORE)

            context_tools = ContextTools(self.context_store, knowledge_path=knowledge_path)
            registry.register_native("update_context", context_tools.update_context, source=ToolSource.CORE)
            registry.register_native("list_context", context_tools.list_context, source=ToolSource.CORE)
            registry.register_native("update_knowledge", context_tools.update_knowledge, source=ToolSource.CORE)
            registry.register_native("list_knowledge", context_tools.list_knowledge, source=ToolSource.CORE)
            registry.register_native("fetch_knowledge", context_tools.fetch_knowledge, source=ToolSource.CORE)


            # 2. Filesystem & Discovery Tools (Priority 2)
            fs_tools = FilesystemTools(self.session_path)
            registry.register_native("ls", fs_tools.ls, source=ToolSource.CORE)
            registry.register_native("read_file", fs_tools.read_file, source=ToolSource.CORE)
            registry.register_native("write_file", fs_tools.write_file, source=ToolSource.CORE)
            # Assuming grep_search and others are in fs_tools or registered similarly
            
            # 3. LangChain Community Tools (Priority 3: Heavy Execution)
            try:
                from langchain_community.tools import ShellTool, YouTubeSearchTool
                from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
                from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
                from langchain_community.utilities import WikipediaAPIWrapper
                
                # shell_exec
                registry.register_langchain_tool(ShellTool())
                
                # web_search
                registry.register_langchain_tool(DuckDuckGoSearchRun())
                
                # wikipedia
                api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
                registry.register_langchain_tool(WikipediaQueryRun(api_wrapper=api_wrapper))
                
                # python_repl - Note: requires langchain_experimental or specific community package
                try:
                    from langchain_experimental.tools import PythonREPLTool
                    registry.register_langchain_tool(PythonREPLTool())
                except ImportError:
                    logger.warning("langchain_experimental not found, skipping PythonREPLTool")

            except ImportError as e:
                logger.warning(f"Failed to load some community tools: {e}")

            # Initialize Guardrails with LLM Policy Generator
            http_client = get_platform_async_http_client()
            policy_gen = LLMPolicyGenerator(
                model_name=self.model_config["model_name"],
                base_url=self.model_config["openai_base_url"],
                http_client=http_client
            )
            # Inject Context Store into Guardrails
            guardrails = GuardrailManager(policy_generator=policy_gen, context_store=self.context_store)

            dispatcher = ToolDispatcher(registry, ProcessSandboxRunner(), guardrails)
            tool_node = AgentToolNode(dispatcher)

            # Result Hook for automated scratch_pad offloading
            result_hook = OffloadingResultHook(knowledge_path)

            # UnitCompiler for recursive spawning
            from ..orch.unit_compiler import UnitCompiler
            unit_compiler = UnitCompiler(
                self.factory, self.mailbox, self.generator,
                dispatcher=dispatcher,
                result_hook=result_hook,
                model_config=self.model_config
            )

            # Fetch tool manifest for prompt injection
            tool_manifest = registry.get_tool_manifest()

            # Resolve Agent Type (Role) from message or defaults
            role = inbox_msg.get("role", AgentRole.SUPERVISOR)

            # Initialize LLM
            llm = ChatOpenAI(
                model=self.model_config["model_name"],
                openai_api_base=self.model_config["openai_base_url"],
                http_async_client=http_client,
                temperature=0,
                max_tokens=10000,
                model_kwargs={
                    "response_format": {"type": "json_object"}
                    }
            )

            # Initialize Unified Orchestrator (v3.0)
            agent_instance = OrchestratorAgent(
                self.factory, 
                self.mailbox, 
                self.generator, 
                llm=llm,
                context_store=self.context_store,
                unit_compiler=unit_compiler,
                tool_manifest=tool_manifest,
                result_hook=result_hook
            )
            graph = agent_instance.build_graph(checkpointer=checkpointer, tool_node=tool_node)
            config = {"configurable": {"thread_id": agent_id}}
            current_state = await graph.aget_state(config)
            
            if not current_state.values:
                initial_state = create_initial_state(
                    agent_id, self.user_id, self.session_id,
                    inbox_path=inbox_path,
                    outbox_path=outbox_path,
                    todo_path=todo_path,
                    role=role
                )
                initial_state["messages"] = [{"role": "user", "content": json.dumps(inbox_msg.get("payload", {}))}]
                await graph.ainvoke(initial_state, config=config)
            else:
                await graph.ainvoke(None, config=config)

            logger.info(f"Work Done.")
            # Clear processed message
            for f in inbox_path.glob("*.json"):
                f.unlink()
