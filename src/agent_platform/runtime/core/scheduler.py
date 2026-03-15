import asyncio
import logging
import aiosqlite
from pathlib import Path
from typing import Dict, Any, Optional, List
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from .workspace import WorkspaceContext
from .mailbox import Mailbox, FilesystemMailboxProvider
from .agent_factory import AgentFactory
from ..agents.supervisor import SupervisorAgent
from ..agents.generator import SystemGeneratorAgent
from ..orch.state import create_initial_state

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
        
        # Infrastructure
        self.factory = AgentFactory(workspace)
        self.mailbox = Mailbox(FilesystemMailboxProvider(self.session_path))
        
        # Configuration for agents
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
        """Main polling loop for the scheduler."""
        logger.info(f"Scheduler started for session {self.session_id}")
        
        while True:
            try:
                await self.tick()
            except Exception as e:
                logger.error(f"Scheduler tick error: {e}", exc_info=True)
            
            await asyncio.sleep(2) # Poll every 2 seconds

    async def tick(self):
        """A single pass over all agents to check for work."""
        agents_root = self.session_path / "agents"
        if not agents_root.exists():
            return

        for agent_dir in agents_root.iterdir():
            if not agent_dir.is_dir():
                continue
            
            agent_id = agent_dir.name
            await self._process_agent(agent_id)

    async def _process_agent(self, agent_id: str):
        """Checks an agent's inbox and triggers execution if a task exists."""
        inbox_msg = self.mailbox.receive(agent_id)
        if not inbox_msg:
            return

        logger.info(f"Scheduler: Triggering execution for agent '{agent_id}'")

        # Resolve paths correctly
        agent_dir = self.workspace.get_agent_dir(self.user_id, self.session_id, agent_id)
        inbox_path = agent_dir / "inbox"
        outbox_path = agent_dir / "outbox"
        knowledge_path = self.session_path / "knowledge"
        todo_path = self.session_path / "todo"

        # 1. Setup Persistence for this agent
        db_path = self.factory.get_agent_db_path(self.user_id, self.session_id, agent_id)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # USE ASYNC SQLITE
        async with aiosqlite.connect(db_path) as conn:
            checkpointer = AsyncSqliteSaver(conn)
            # Initialize checkpointer
            await checkpointer.setup()
            
            # 2. Initialize Agent
            agent_instance = SupervisorAgent(
                self.factory, self.mailbox, self.generator, 
                model_name=self.model_config["model_name"],
                base_url=self.model_config["openai_base_url"]
            )
            
            # Setup Tooling for this agent
            from .dispatcher import ToolRegistry, ToolDispatcher, ToolSource
            from ..core.sandbox import ProcessSandboxRunner
            from ..core.guardrails import GuardrailManager
            from ..storage.todo_tool import TODOTool
            
            registry = ToolRegistry(self.session_path)
            todo_tool = TODOTool(self.session_path)
            
            # Register native TODO tools
            registry.register_native("add_task", todo_tool.add_task)
            registry.register_native("list_tasks", todo_tool.list_tasks)
            registry.register_native("update_task_status", todo_tool.update_task_status)
            
            sandbox = ProcessSandboxRunner()
            guardrails = GuardrailManager()
            dispatcher = ToolDispatcher(registry, sandbox, guardrails)
            
            graph = agent_instance.build_graph(checkpointer=checkpointer)
            
            # 3. Resolve State
            config = {"configurable": {"thread_id": agent_id}}
            current_state = await graph.aget_state(config)
            
            if not current_state.values:
                # New agent - initialize from the inbox message
                initial_state = create_initial_state(
                    agent_id, self.user_id, self.session_id,
                    inbox_path, outbox_path, knowledge_path, todo_path
                )
                initial_state["messages"] = [{"role": "user", "content": str(inbox_msg.get("payload", {}))}]
                await graph.ainvoke(initial_state, config=config)
            else:
                # Existing agent - resume
                await graph.ainvoke(None, config=config)
            
            # 4. Cleanup Inbox
            for f in inbox_path.glob("*.json"):
                f.unlink()
