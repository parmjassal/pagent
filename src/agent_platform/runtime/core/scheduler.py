import asyncio
import logging
import aiosqlite
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from .workspace import WorkspaceContext
from .mailbox import Mailbox, FilesystemMailboxProvider
from .agent_factory import AgentFactory
from ..agents.supervisor import SupervisorAgent
from ..agents.generator import SystemGeneratorAgent
from ..orch.state import create_initial_state
from .dispatcher import ToolRegistry, ToolDispatcher, ToolSource
from .sandbox import ProcessSandboxRunner
from .guardrails import GuardrailManager
from .todo import TODOManager, ScopedTask
from ..storage.todo_tool import TODOTool
from .tools.filesystem import FilesystemTools
from .context_store import FilesystemContextStore
from ..storage.context_tool import ContextTools

logger = logging.getLogger(__name__)

class AutonomousScheduler:
    """
    The background engine that monitors mailboxes and triggers agent execution.
    Turns the static skeleton into a self-driving platform.
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
        
        self.factory = AgentFactory(workspace)
        self.mailbox = Mailbox(FilesystemMailboxProvider(self.session_path))
        
        self.model_config = {
            "model_name": model_name,
            "openai_base_url": openai_base_url
        }

        self.generator = SystemGeneratorAgent(
            workspace=workspace,
            model_name=model_name,
            base_url=openai_base_url
        )

    async def run_forever(self):
        logger.info(f"Scheduler started for session {self.session_id}")
        while True:
            try:
                await self.tick()
            except Exception as e:
                logger.error(f"Scheduler tick error: {e}", exc_info=True)
            await asyncio.sleep(2)

    async def tick(self):
        """Recursively discovers and processes agents."""
        agents_root = self.session_path / "agents"
        if not agents_root.exists():
            return

        # Use os.walk to find all agent directories (those with an 'inbox')
        for root, dirs, files in os.walk(agents_root):
            root_path = Path(root)
            if (root_path / "inbox").exists():
                # The agent_id is the relative path from agents_root
                agent_id = str(root_path.relative_to(agents_root))
                await self._process_agent(agent_id)

    async def _process_agent(self, agent_id: str):
        inbox_msg = self.mailbox.receive(agent_id)
        if not inbox_msg:
            return

        logger.info(f"Scheduler: Triggering execution for agent '{agent_id}'")

        agent_dir = self.workspace.get_agent_dir(self.user_id, self.session_id, agent_id)
        inbox_path = agent_dir / "inbox"
        outbox_path = agent_dir / "outbox"
        knowledge_path = self.session_path / "knowledge"
        todo_path = agent_dir / "todo" # Corrected: Agent-level TODO

        db_path = self.factory.get_agent_db_path(self.user_id, self.session_id, agent_id)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(db_path) as conn:
            checkpointer = AsyncSqliteSaver(conn)
            await checkpointer.setup()
            
            registry = ToolRegistry(self.session_path)
            
            # 1. TODO Tools
            todo_tool = TODOTool(self.session_path)
            registry.register_native("add_task", todo_tool.add_task)
            registry.register_native("list_tasks", todo_tool.list_tasks)
            registry.register_native("update_task_status", todo_tool.update_task_status)

            # 2. Filesystem tools
            fs_tools = FilesystemTools(self.session_path)
            registry.register_native("ls", fs_tools.ls)
            registry.register_native("read_file", fs_tools.read_file)
            registry.register_native("write_file", fs_tools.write_file)
            
            # 3. Context Store & Tools (Hierarchical)
            context_store = FilesystemContextStore(self.session_path)
            context_tools = ContextTools(context_store)
            registry.register_native("list_context", context_tools.list_context)
            registry.register_native("read_context", context_tools.read_context)
            registry.register_native("update_context", context_tools.update_context)
            registry.register_native("search_context", context_tools.search_context)

            sandbox = ProcessSandboxRunner()
            guardrails = GuardrailManager() 
            dispatcher = ToolDispatcher(registry, sandbox, guardrails)
            
            agent_instance = SupervisorAgent(
                self.factory, self.mailbox, self.generator, 
                model_name=self.model_config["model_name"],
                base_url=self.model_config["openai_base_url"]
            )
            
            graph = agent_instance.build_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": agent_id}}
            current_state = await graph.aget_state(config)
            
            if not current_state.values:
                initial_state = create_initial_state(
                    agent_id, self.user_id, self.session_id,
                    inbox_path, outbox_path, knowledge_path, todo_path
                )
                initial_state["messages"] = [{"role": "user", "content": str(inbox_msg.get("payload", {}))}]
                await graph.ainvoke(initial_state, config=config)
            else:
                await graph.ainvoke(None, config=config)
            
            for f in inbox_path.glob("*.json"):
                f.unlink()
