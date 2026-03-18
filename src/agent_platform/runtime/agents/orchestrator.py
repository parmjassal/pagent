import logging
import json
import secrets
from typing import Dict, Any, List, Optional, Union
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage # NEW: ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers.retry import RetryWithErrorOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

#  FIX: correct import path
from langchain_core.prompt_values import ChatPromptValue

from ..orch.state import AgentState, AgentRole
from ..orch.models import PlanningResult, ExecutionStrategy, SubAgentTask, Action
from ..orch.quota import SessionQuota, update_quota
from ..core.agent_factory import AgentFactory
from ..core.mailbox import Mailbox
from ..core.todo import TODOManager, ScopedTask, TaskStatus, TaskType
from ..core.context_store import ContextStore
from ..core.parser import robust_json_parser
from .generator import SystemGeneratorAgent, TaskType as GenTaskType
from ..orch.result_hook import ResultHook

logger = logging.getLogger(__name__)

#  FIX: normalize dict → BaseMessage
def _normalize_messages(messages):
    normalized = []
    for m in messages:
        if isinstance(m, BaseMessage):
            normalized.append(m)
        elif isinstance(m, dict):
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                normalized.append(HumanMessage(content=content))
            elif role == "assistant":
                normalized.append(AIMessage(content=content))
            elif role == "tool": # NEW: Handle dict-based tool messages
                # tool_call_id is essential for ToolMessage
                tid = m.get("tool_call_id") or secrets.token_hex(4)
                normalized.append(ToolMessage(content=content, tool_call_id=tid))
            else:
                normalized.append(HumanMessage(content=content))
        else:
            normalized.append(HumanMessage(content=str(m)))
    return normalized


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
        logger.info(f"Agent {state['agent_id']} entering planner_node.")
        
        pydantic_parser = PydanticOutputParser(pydantic_object=PlanningResult)
        
        retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=pydantic_parser,
            llm=self.llm,
            max_retries=3
        )

        session_path = self.agent_factory.workspace.get_session_dir(state["user_id"], state["session_id"])
        
        template_name = "supervisor_decompose.txt" if state["role"] == AgentRole.SUPERVISOR else "worker_reasoning.txt"
        template_path = session_path / "prompts" / template_name
        
        system_instruction = "Plan the next batch of tasks to achieve the goal."
        if template_path.exists():
            system_instruction = template_path.read_text()

        if self.tool_manifest:
            system_instruction = f"""{system_instruction}

## Available Tools:
{self.tool_manifest}"""
        
        todo_mgr = TODOManager(state["todo_path"].parent)
        all_tasks = todo_mgr.list_tasks()
        if all_tasks:
            todo_summary = "\n".join([f"- [{t.status}] {t.task_id}: {t.title} ({t.type})" for t in all_tasks])
            system_instruction = f"""{system_instruction}

## Current Work Order (TODO List):
{todo_summary}
"""

        #  FIX: normalize messages
        prompt_messages = [
            SystemMessage(content=system_instruction),
            *_normalize_messages(state["messages"])
        ]
        
        try:
            logger.debug(f"Invoking LLM: {prompt_messages}")
            raw_response = await self.llm.ainvoke(prompt_messages)
            logger.debug(f"Planning Raw Response: {raw_response}")

            # Normalize completion
            if isinstance(raw_response, list):
                completion = "\n".join(
                    m.content if hasattr(m, "content") else str(m)
                    for m in raw_response
                )
            else:
                completion = getattr(raw_response, "content", str(raw_response))

            # ---------------------------------------------------------
            # ✅ FIRST: Fast robust JSON parse
            # ---------------------------------------------------------
            parsed_dict = robust_json_parser(completion)

            if parsed_dict:
                try:
                    result: PlanningResult = PlanningResult.model_validate(parsed_dict)
                    logger.debug(f"Parsed via robust_json_parser (fast path) {result}")
                except Exception as e:
                    logger.debug(f"Fast parse failed validation, falling back: {e}")
                    result = None
            else:
                result = None

            # ---------------------------------------------------------
            # 🔁 FALLBACK: Retry parser ONLY if needed
            # ---------------------------------------------------------
            if result is None:
                prompt_value = ChatPromptValue(messages=prompt_messages)

                result: PlanningResult = await retry_parser.aparse_with_prompt(
                    completion,
                    prompt_value
                )

                logger.debug("Parsed via retry_parser (fallback)")

            # ✅ common success path
            logger.info(f"Planner turn complete with {len(result.action_sequence)} actions.")

        except Exception as e:
            logger.error(f"Planning failed after retries: {e}")
            return {"messages": [{"role": "user", "content": f"[System] Planning Error: {e}"}]}

        metadata_update = {"thought_process": result.thought_process}
        
        todo_mgr = TODOManager(state["todo_path"].parent)
        new_task_ids = []
        
        for action in result.action_sequence:
            if action.strategy == ExecutionStrategy.FINISH:
                tid = todo_mgr.add_task(ScopedTask(
                    title="Finish",
                    description=action.final_answer or "Goal achieved.",
                    type=TaskType.FINISH,
                    payload={"final_answer": action.final_answer}
                ))
                new_task_ids.append(tid)
            
            elif action.strategy == ExecutionStrategy.DECOMPOSE:
                if action.sub_tasks:
                    for st in action.sub_tasks:
                        tid = todo_mgr.add_task(ScopedTask(
                            title=f"Sub-agent: {st.agent_id}",
                            description=st.instructions,
                            type=TaskType.AGENT,
                            assigned_to=st.agent_id,
                            metadata={"role": st.role}
                        ))
                        new_task_ids.append(tid)

            elif action.strategy in (ExecutionStrategy.TOOL_USE, ExecutionStrategy.AUTHORIZE):
                tool_args = action.args or {}
                if action.strategy == ExecutionStrategy.AUTHORIZE and 'content' in tool_args and not isinstance(tool_args['content'], str):
                    tool_args['content'] = json.dumps(tool_args['content'])

                tid = todo_mgr.add_task(ScopedTask(
                    title=f"{action.strategy.capitalize()}: {action.name}",
                    description=f"Invoke {action.name}",
                    type=TaskType.TOOL,
                    payload={"name": action.name, "args": tool_args}
                ))
                new_task_ids.append(tid)

        return {
            "messages": [{"role": "assistant", "content": result.thought_process, "tool_calls":[]}],
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
        
        if task.type == TaskType.FINISH:
             # Mark as COMPLETED immediately as there is no executor for it
             todo_mgr.update_status(task.task_id, TaskStatus.COMPLETED)
             final_answer = task.payload.get("final_answer", "Goal achieved.")
             processed = self.result_hook.process_result(state["agent_id"], final_answer) if self.result_hook else {"type": "inline", "content": final_answer}
             return {
                 "final_result": processed,
                 "metadata": {"next_task": None}
             }

        # Mark as IN_PROGRESS
        todo_mgr.update_status(task.task_id, TaskStatus.IN_PROGRESS)
        
        # Metadata setup
        meta_update = {"next_task": task.model_dump()}
        msg_update = []
        
        tool_call_id = f"call_{secrets.token_hex(4)}" # Unified tool_call_id generation
        
        if task.type == TaskType.TOOL:
            tool_call = task.payload
            meta_update["next_tool_call"] = {
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call_id
            }
            # Inject assistant message for strict APIs
            msg_update.append(AIMessage(
                content=f"Executing tool {tool_call['name']} for task {task.task_id}...",
                tool_calls=[{
                    "id": tool_call_id,
                    "name": tool_call["name"],
                    "args": tool_call["args"]
                }]
            ))
        elif task.type == TaskType.AGENT: # Handle AGENT type as a tool_call
            meta_update["next_tool_call"] = {
                "name": "delegate_to_agent", # A generic tool name for delegation
                "args": {"agent_id": task.assigned_to, "instructions": task.description},
                "id": tool_call_id
            }

            msg_update.append(AIMessage(
                content=f"Delegating to sub-agent {task.assigned_to} for task {task.task_id}...", 
                tool_calls=[{
                    "id": tool_call_id,
                    "name": "delegate_to_agent",
                    "args": {
                        "agent_id": task.assigned_to, 
                        "instructions": task.description
                    }
                }]
            ))
            
        return {"metadata": meta_update, "messages": msg_update }

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
                "messages": [{"role": "user", "content": prompt}],
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
        
        # Unified tool_call_id tracking
        next_tool_call = state["metadata"].get("next_tool_call")
        tool_call_id = next_tool_call.get("id") if next_tool_call else None
        tool_name = next_tool_call.get("name") if next_tool_call else "delegate_to_agent"
        
        # Result can come from ToolNode (as a new message) or ExecutorNode (as metadata)
        result = state["metadata"].get("task_result")
        error = state["metadata"].get("task_error")
        
        # If result is not in metadata, it might be the last message (from ToolNode)
        if not result and not error and state["messages"]:
            # ToolNode might have added multiple messages (result + log)
            # We look for the one with role 'tool' or 'user' containing the result
            for msg in reversed(state["messages"]):
                if isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content")
                else:
                    role = getattr(msg, "role", None)
                    content = getattr(msg, "content", None)
                
                if role in ("tool", "user"):
                    result = content
                    break

        status = TaskStatus.FAILED if error else TaskStatus.COMPLETED
        final_val = error if error else result
        
        todo_mgr.update_status(task_id, status, result={"output": final_val})

        # If failure, clear remaining pending tasks for this batch to force re-planning
        if status == TaskStatus.FAILED:
            pending = todo_mgr.list_tasks(status=TaskStatus.PENDING)
            for p_task in pending:
                todo_mgr.update_status(p_task.task_id, TaskStatus.FAILED, result={"output": "Aborted due to previous failure."})
        
        # Determine if we need to add a message to the state
        msg_update = []
        is_already_present = False
        
        if tool_call_id and state["messages"]:
            # Check if ToolNode already added a 'tool' message for this ID
            for msg in reversed(state["messages"]):
                if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
                    is_already_present = True
                    break
                elif hasattr(msg, "tool_call_id") and getattr(msg, "tool_call_id") == tool_call_id:
                    is_already_present = True
                    break

        if tool_call_id and not is_already_present:
            # AGENT tasks (handled by executor_node) need a 'tool' message to match the dispatcher's AIMessage
            msg_update.append({
                "role": "tool", 
                "name": tool_name, 
                "content": str(final_val), 
                "tool_call_id": tool_call_id
            })
        
        # Always add a system log message for clarity in the UI/logs
        #msg_update.append({"role": "user", "content": f"[System] Task {task_id} finished: {str(final_val)[:200]}..."})
        
        # Clean up transient metadata
        return {
            "messages": msg_update,
            "metadata": {"next_task": None, "task_result": None, "task_error": None, "next_tool_call": None}
        }

    def _route_post_dispatch(self, state: AgentState) -> str:
        if state["metadata"].get("next_task"):
            task = state["metadata"]["next_task"]
            return "executor" if task["type"] == TaskType.AGENT else "tools"
        if isinstance(state["messages"][-1], AIMessage):
            state["messages"].append({"role": "user", "content": "Continue reasoning loop."})    
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
