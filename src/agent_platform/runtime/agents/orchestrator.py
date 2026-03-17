import logging
import json
import secrets
from typing import Dict, Any, List, Optional, Union
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..orch.state import AgentState, AgentRole
from ..orch.models import PlanningResult, ExecutionStrategy, SubAgentTask, ToolCall
from ..orch.quota import SessionQuota, update_quota
from ..core.agent_factory import AgentFactory
from ..core.mailbox import Mailbox
from ..core.todo import TODOManager, ScopedTask, TaskStatus, TaskType
from ..core.context_store import ContextStore
from ..core.parser import robust_json_parser
from .generator import SystemGeneratorAgent, TaskType as GenTaskType
from ..orch.result_hook import ResultHook

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """
    Unified Orchestrator agent that follows a Batch-Sequential execution model.
    Planner -> Dispatcher -> Executor -> Collector -> Router loop.
    """
    def __init__(
        self, 
        agent_factory: AgentFactory, 
        mailbox: Mailbox, 
        generator: SystemGeneratorAgent,
        llm: Any,
        context_store: Optional[ContextStore] = None,
        unit_compiler: Optional[Any] = None,
        tool_manifest: Optional[str] = None,
        result_hook: Optional[ResultHook] = None
    ):
        self.agent_factory = agent_factory
        self.mailbox = mailbox
        self.generator = generator
        self.llm = llm
        self.context_store = context_store
        self.unit_compiler = unit_compiler
        self.tool_manifest = tool_manifest
        self.result_hook = result_hook

    async def planner_node(self, state: AgentState) -> AgentState:
        """Thinker: Reviews history and writes a batch of tasks to the TODO list."""
        logger.info(f"Agent {state['agent_id']} entering planner_node.")
        
        # Resolve Template
        session_path = self.agent_factory.workspace.get_session_dir(state["user_id"], state["session_id"])
        
        # Use role-specific template or default orchestrator
        template_name = "supervisor_decompose.txt" if state["role"] == AgentRole.SUPERVISOR else "worker_reasoning.txt"
        template_path = session_path / "prompts" / template_name
        
        system_instruction = "Plan the next batch of tasks to achieve the goal."
        if template_path.exists():
            system_instruction = template_path.read_text()

        # Inject Tool Manifest
        if self.tool_manifest:
            system_instruction = f"{system_instruction}\n\n## Available Tools:\n{self.tool_manifest}"
        
        # Inject current TODO state for context
        todo_mgr = TODOManager(state["todo_path"].parent)
        all_tasks = todo_mgr.list_tasks()
        if all_tasks:
            todo_summary = "\n".join([f"- [{t.status}] {t.task_id}: {t.title} ({t.type})" for t in all_tasks])
            system_instruction = f"{system_instruction}\n\n## Current Work Order (TODO List):\n{todo_summary}\n"

        prompt = [
            SystemMessage(content=system_instruction),
            *state["messages"]
        ]
        
        # Use with_structured_output for robust, model-native JSON parsing
        structured_llm = self.llm.with_structured_output(PlanningResult)
        
        try:
            result = await structured_llm.ainvoke(prompt)
            logger.debug(f"Planning Result is {result}")
            logger.info(f"Planner strategy: {result.strategy}")
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {"messages": [{"role": "user", "content": f"[System] Planning Error: {e}"}]}

        # Handle Plan
        metadata_update = {"thought_process": result.thought_process, "strategy": result.strategy}
        
        if result.strategy == ExecutionStrategy.FINISH:
            # Handle both PlanningResult's implied finish and potential future final_answer
            final_answer = result.thought_process or "Goal achieved."
            # If the model used a WorkerResult-like structure (which we can allow via loose parsing)
            if hasattr(result, "final_answer") and result.final_answer:
                final_answer = result.final_answer
            
            processed = self.result_hook.process_result(state["agent_id"], final_answer) if self.result_hook else {"type": "inline", "content": final_answer}
            return {
                "messages": [{"role": "assistant", "content": result.thought_process}],
                "final_result": processed,
                "metadata": metadata_update,
                "node_counts": {"plan": 1}
            }

        # Convert PlanningResult into ScopedTasks in TODO
        todo_mgr = TODOManager(state["todo_path"].parent)
        new_task_ids = []
        
        if result.strategy == ExecutionStrategy.DECOMPOSE and result.sub_tasks:
            for st in result.sub_tasks:
                tid = todo_mgr.add_task(ScopedTask(
                    title=f"Sub-agent: {st.agent_id}",
                    description=st.instructions,
                    type=TaskType.AGENT,
                    assigned_to=st.agent_id,
                    metadata={"role": st.role}
                ))
                new_task_ids.append(tid)
        
        elif result.strategy == ExecutionStrategy.TOOL_USE and result.tool_call:
            tid = todo_mgr.add_task(ScopedTask(
                title=f"Tool: {result.tool_call.name}",
                description=f"Invoke {result.tool_call.name}",
                type=TaskType.TOOL,
                payload={"name": result.tool_call.name, "args": result.tool_call.args}
            ))
            new_task_ids.append(tid)

        return {
            "messages": [{"role": "assistant", "content": result.thought_process}],
            "metadata": metadata_update,
            "node_counts": {"plan": 1}
        }

    async def dispatcher_node(self, state: AgentState) -> AgentState:
        """Pops the next PENDING task from TODO."""
        todo_mgr = TODOManager(state["todo_path"].parent)
        pending = [t for t in todo_mgr.list_tasks() if t.status == TaskStatus.PENDING]
        
        if not pending:
            return {"metadata": {"next_task": None}}
        
        # Sequential: Pick the first one
        task = pending[0]
        # Mark as IN_PROGRESS
        todo_mgr.update_status(task.task_id, TaskStatus.IN_PROGRESS)
        
        # Metadata setup
        meta_update = {"next_task": task.model_dump()}
        msg_update = []
        
        if task.type == TaskType.TOOL:
            tool_call = task.payload
            tool_call_id = f"call_{secrets.token_hex(4)}"
            meta_update["next_tool_call"] = {
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call_id
            }
            # Inject assistant message for strict APIs
            msg_update.append({
                "role": "assistant", 
                "content": f"Executing task {task.task_id}...",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_call["name"], "arguments": json.dumps(tool_call["args"])}
                }]
            })
            
        return {"metadata": meta_update, "messages": msg_update}

    async def executor_node(self, state: AgentState) -> AgentState:
        """Dispatches to SpawningNode (AGENT type). TOOL type is handled by ToolNode."""
        task_data = state["metadata"].get("next_task")
        if not task_data or task_data["type"] != TaskType.AGENT:
            return state
        
        task = ScopedTask.model_validate(task_data)
        
        # AGENT type: Prepare for spawning
        # We reuse the logic from Supervisor but streamlined
        state["metadata"]["current_task_instructions"] = task.description
        state["next_steps"] = [task.assigned_to] # Point to assigned agent
        
        # Role mapping from task metadata
        next_role = task.metadata.get("role", AgentRole.WORKER)
        
        # Step 1: Generate Prompt
        prompt_res = await self.generator.generate_node(state, task_type=GenTaskType.PROMPT)
        prompt = prompt_res.get("generated_output", "Process task.")
        
        # Step 2: Spawn
        sub_agent_id = f"{state['agent_id']}/{task.assigned_to}"
        new_agent = self.agent_factory.create_agent(
            user_id=state["user_id"], session_id=state["session_id"],
            agent_id=sub_agent_id, current_quota=state["quota"], parent_depth=state["current_depth"]
        )
        
        if not new_agent:
            return {"metadata": {"task_error": "Quota reached"}, "quota": SessionQuota(agent_count=0)}

        # Recursive if UnitCompiler exists
        if self.unit_compiler:
            child_graph = self.unit_compiler.compile_unit(next_role)
            agent_dir = self.agent_factory.workspace.get_agent_dir(state["user_id"], state["session_id"], sub_agent_id)
            child_state = {
                **state, "agent_id": sub_agent_id, "role": next_role,
                "messages": [{"role": "system", "content": prompt}],
                "inbox_path": agent_dir/"inbox", "outbox_path": agent_dir/"outbox", "todo_path": agent_dir/"todo",
                "current_depth": state["current_depth"] + 1, "final_result": None, "next_steps": [], "node_counts": {}
            }
            child_final = await child_graph.ainvoke(child_state)
            result = child_final.get("final_result", "Completed.")
            return {
                "metadata": {"task_result": f"Sub-agent {sub_agent_id} returned: {result}"},
                "quota": SessionQuota(agent_count=1)
            }

        else:
            # Async Mailbox
            self.mailbox.send(sub_agent_id, {
                "id": f"task_{task.task_id}", "sender": state["agent_id"],
                "system_prompt": prompt, "payload": {"instructions": task.description}, "role": next_role
            })
            return {
                "metadata": {"task_result": f"Spawned {sub_agent_id} via Mailbox"},
                "quota": SessionQuota(agent_count=1)
            }

    async def collector_node(self, state: AgentState) -> AgentState:
        """Receives result and updates TODO list."""
        task_data = state["metadata"].get("next_task")
        if not task_data:
            return state
        
        task_id = task_data["task_id"]
        todo_mgr = TODOManager(state["todo_path"].parent)
        
        # Result can come from ToolNode (as a new message) or ExecutorNode (as metadata)
        result = state["metadata"].get("task_result")
        error = state["metadata"].get("task_error")
        
        # If result is not in metadata, it might be the last message (from ToolNode)
        if not result and not error:
            last_msg = state["messages"][-1]
            if isinstance(last_msg, dict):
                role = last_msg.get("role")
                content = last_msg.get("content")
            else:
                role = getattr(last_msg, "role", None)
                content = getattr(last_msg, "content", None)
            
            if role in ("tool", "user"):
                result = content

        status = TaskStatus.FAILED if error else TaskStatus.COMPLETED
        final_val = error if error else result
        
        todo_mgr.update_status(task_id, status, result={"output": final_val})
        
        # Clean up transient metadata
        return {
            "messages": [{"role": "user", "content": f"[System] Task {task_id} finished: {str(final_val)[:200]}..."}],
            "metadata": {"next_task": None, "task_result": None, "task_error": None, "next_tool_call": None}
        }

    def _route_post_dispatch(self, state: AgentState) -> str:
        if state["metadata"].get("next_task"):
            task = state["metadata"]["next_task"]
            return "executor" if task["type"] == TaskType.AGENT else "tools"
        return "planner"

    def build_graph(self, checkpointer: Optional[BaseCheckpointSaver] = None, tool_node: Optional[Any] = None) -> Any:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("dispatcher", self.dispatcher_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("collector", self.collector_node)
        workflow.add_node("tools", tool_node or (lambda s: s)) # Placeholder if not provided
        
        workflow.set_entry_point("planner")
        
        workflow.add_conditional_edges("planner", lambda s: END if s.get("final_result") else "dispatcher")
        workflow.add_conditional_edges("dispatcher", self._route_post_dispatch)
        
        workflow.add_edge("executor", "collector")
        workflow.add_edge("tools", "collector")
        workflow.add_edge("collector", "dispatcher") # Sequential loop
        
        return workflow.compile(checkpointer=checkpointer)
