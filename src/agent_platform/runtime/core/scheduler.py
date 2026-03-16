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
from .guardrails import GuardrailManager
from .todo import TODOManager
from ..agents.supervisor import SupervisorAgent
from ..agents.worker import WorkerAgent
from ..agents.generator import SystemGeneratorAgent
from ..orch.state import create_initial_state, AgentRole
from ..orch.tool_node import AgentToolNode
from ..core.tools.filesystem import FilesystemTools
from ..core.context_store import FilesystemContextStore
from ..storage.context_tool import ContextTools
from ..storage.todo_tool import TODOTool
from ..core.http_client import get_platform_http_client

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
        # Shared generator for the session
        self.generator = SystemGeneratorAgent(
            model_name=model_name,
            base_url=openai_base_url,
            workspace=workspace
        )

    async def run_forever(self):
        """Main loop of the scheduler."""
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
            
            context_store = FilesystemContextStore(self.session_path)
            context_tools = ContextTools(context_store)
            registry.register_native("update_context", context_tools.update_context)

            dispatcher = ToolDispatcher(registry, ProcessSandboxRunner(), GuardrailManager())
            tool_node = AgentToolNode(dispatcher)

            # Resolve Agent Type (Role) from message or defaults
            role = inbox_msg.get("role", AgentRole.WORKER)
            
            # Initialize LLM
            http_client = get_platform_http_client()
            llm = ChatOpenAI(
                model=self.model_config["model_name"],
                openai_api_base=self.model_config["openai_base_url"],
                http_client=http_client,
                temperature=0
            )

            if role == AgentRole.SUPERVISOR:
                agent_instance = SupervisorAgent(
                    self.factory, self.mailbox, self.generator, llm=llm
                )
                graph = agent_instance.build_graph(checkpointer=checkpointer, tool_node=tool_node)
            else:
                agent_instance = WorkerAgent(tool_node, llm=llm)
                graph = agent_instance.build_graph(checkpointer=checkpointer)

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
            
            # Clear processed message
            for f in inbox_path.glob("*.json"):
                f.unlink()
