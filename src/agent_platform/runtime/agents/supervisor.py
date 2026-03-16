import logging
import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..orch.state import AgentState, AgentRole
from ..orch.models import PlanningResult, ExecutionStrategy, SubAgentTask
from ..orch.quota import SessionQuota
from ..core.agent_factory import AgentFactory
from ..core.mailbox import Mailbox
from ..core.todo import TODOManager, ScopedTask
from ..core.context_store import ContextStore
from ..core.parser import robust_json_parser
from .generator import SystemGeneratorAgent, TaskType

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """
    Supervisor agent responsible for decomposing tasks and spawning sub-agents.
    Now supports both TOOL_USE and DECOMPOSE strategies.
    """
    def __init__(
        self, 
        agent_factory: AgentFactory, 
        mailbox: Mailbox, 
        generator: SystemGeneratorAgent,
        llm: Any,
        context_store: Optional[ContextStore] = None,
        unit_compiler: Optional[Any] = None, # For recursive execution
        tool_manifest: Optional[str] = None,
        result_hook: Optional[Any] = None
    ):
        self.agent_factory = agent_factory
        self.mailbox = mailbox
        self.generator = generator
        self.llm = llm
        self.context_store = context_store
        self.unit_compiler = unit_compiler
        self.tool_manifest = tool_manifest
        self.result_hook = result_hook
        self.parser = JsonOutputParser(pydantic_object=PlanningResult)

    async def planning_node(self, state: AgentState) -> AgentState:
        """Determines the execution strategy based on current state."""
        logger.info(f"Agent {state['agent_id']} entering planning_node. Node counts: {state['node_counts']}")
        
        # 0. Intent Persistence: If this is the first planning step, record the intent
        if state["node_counts"].get("plan", 0) == 0 and self.context_store:
            initial_msg = ""
            for m in state["messages"]:
                if isinstance(m, dict) and m.get("role") == "user":
                    initial_msg = str(m.get("content", ""))
                    break
            
            if initial_msg:
                self.context_store.update_fact(
                    state["agent_id"], 
                    "initial_intent", 
                    f"The high-level goal for this session is: {initial_msg}",
                    state["role"]
                )

        # 1. Clear previous next_steps when re-planning to allow pivoting
        # next_steps will be repopulated if the LLM still wants them
        state["next_steps"] = []

        # 2. Resolve Template
        session_path = self.agent_factory.workspace.get_session_dir(state["user_id"], state["session_id"])
        template_path = session_path / "prompts" / "supervisor_decompose.txt"
        
        system_instruction = "Decompose the task."
        if template_path.exists():
            system_instruction = template_path.read_text()

        # 3. Inject Context (Tools and TODO history)
        if self.tool_manifest:
            system_instruction = f"{system_instruction}\n\n{self.tool_manifest}"
        
        # Inject TODO state for execution history
        todo_mgr = TODOManager(state["todo_path"].parent)
        all_tasks = todo_mgr.list_tasks()
        if all_tasks:
            todo_summary = "\n".join([f"- [{t.status}] {t.assigned_to}: {t.title}" for t in all_tasks])
            system_instruction = f"{system_instruction}\n\n## Execution History (TODO List):\n{todo_summary}\n"

        # 4. Invoke LLM
        prompt = [
            SystemMessage(content=system_instruction),
            *state["messages"]
        ]
        
        try:
            logger.debug(f"Prompt is {prompt}")
            response = await self.llm.ainvoke(prompt)
            logger.debug(f"Response from llm is {response}")
            # Robust Parsing
            if hasattr(response, "strategy"):
                result = response
            elif hasattr(response, "content"):
                # Handle AIMessage
                try:
                    parsed = robust_json_parser(response.content)
                    result = PlanningResult.model_validate(parsed)
                except Exception as e:
                    logger.error(f"Planning failed to parse JSON: {response.content}")
                    # Corrective message back to the LLM
                    return {
                        "messages": [
                            {"role": "assistant", "content": response.content},
                            {"role": "user", "content": f"[System] Error: Your response could not be parsed as JSON. You MUST return ONLY a valid JSON object conforming to the schema. Error detail: {e}"}
                        ],
                        "node_counts": {"plan": 1}
                    }
            else:
                # Fallback for dicts
                result = PlanningResult.model_validate(response)
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {"messages": [{"role": "user", "content": f"[System] Planning Error: {e}"}], "metadata": {"strategy": ExecutionStrategy.FINISH}}

        metadata_update = {
            "strategy": result.strategy,
            "thought_process": result.thought_process
        }

        if result.strategy == ExecutionStrategy.DECOMPOSE and result.sub_tasks:
            next_steps = []
            existing_agent_ids = [t.assigned_to for t in all_tasks]
            
            for task_def in result.sub_tasks:
                # Only add if not already in TODO or completed
                if task_def.agent_id not in existing_agent_ids:
                    todo_mgr.add_task(ScopedTask(
                        title=f"Task for {task_def.agent_id}",
                        description=task_def.instructions,
                        assigned_to=task_def.agent_id,
                        metadata={"role": task_def.role}
                    ))
                    next_steps.append(task_def.agent_id)
                else:
                    logger.info(f"Skipping redundant task for {task_def.agent_id}")
            
            # Store sub-tasks details in metadata for the generator node
            metadata_update["sub_tasks_config"] = [t.model_dump() for t in result.sub_tasks]
            
            if not next_steps:
                # If all suggested tasks were redundant, ask LLM to re-evaluate or finish
                return {
                    "messages": [{"role": "user", "content": "[System] All proposed sub-tasks are already in progress or completed. Please provide new tasks or finish."}],
                    "node_counts": {"plan": 1}
                }

            return {
                "messages": [{"role": "assistant", "content": result.thought_process}],
                "next_steps": next_steps,
                "metadata": metadata_update,
                "node_counts": {"plan": 1}
            }
        
        elif result.strategy == ExecutionStrategy.TOOL_USE and result.tool_call:
            # Capture ID from AIMessage if it was a real tool call format
            tool_call_id = None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_call_id = response.tool_calls[0].get("id")
            
            tc_dump = result.tool_call.model_dump()
            if tool_call_id:
                tc_dump["id"] = tool_call_id

            metadata_update["next_tool_call"] = tc_dump

            assistant_msg = {"role": "assistant", "content": result.thought_process}
            if tool_call_id:
                # If we use tool_call_id, we MUST provide tool_calls in the assistant message 
                # to satisfy strict API schemas (e.g. ModelArts, some OpenAI versions).
                assistant_msg["tool_calls"] = [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": result.tool_call.name,
                        "arguments": json.dumps(result.tool_call.args)
                    }
                }]

            return {
                "messages": [assistant_msg],
                "metadata": metadata_update,
                "next_steps": [],
                "node_counts": {"plan": 1}
            }

        # Finish strategy: Provide a final result
        final_answer = result.thought_process or "All tasks completed."
        if self.result_hook:
            processed_result = self.result_hook.process_result(state["agent_id"], final_answer)
        else:
            processed_result = {"type": "inline", "content": final_answer}

        return {
            "messages": [{"role": "assistant", "content": result.thought_process}],
            "metadata": metadata_update,
            "next_steps": [],
            "final_result": processed_result,
            "node_counts": {"plan": 1}
        }

    async def generate_prompt_node(self, state: AgentState) -> Dict[str, Any]:
        """Uses SystemGeneratorAgent to build a prompt for the current next_step."""
        if not state["next_steps"]:
            return {}

        current_sub_id = state["next_steps"][0]
        
        # Find instruction for this specific sub-agent from metadata
        sub_tasks = state.get("metadata", {}).get("sub_tasks_config", [])
        instruction = next((t["instructions"] for t in sub_tasks if t["agent_id"] == current_sub_id), "Process task.")
        
        # Inject instruction into metadata for generator to pick up
        state["metadata"]["current_task_instructions"] = instruction
        
        res = await self.generator.generate_node(state, task_type=TaskType.PROMPT)
        return res

    async def spawning_node(self, state: AgentState) -> AgentState:
        """Creates the child workspace and sends the initial message."""
        if not state["next_steps"]:
            return state

        # Prepare IDs
        raw_sub_id = state["next_steps"][0]
        # Use hierarchical ID if not already prefixed
        if "/" not in raw_sub_id:
            sub_agent_id = f"{state['agent_id']}/{raw_sub_id}"
        else:
            sub_agent_id = raw_sub_id
        
        prompt = state.get("generated_output")
        # Find role for this specific sub-agent
        sub_tasks = state.get("metadata", {}).get("sub_tasks_config", [])
        next_role = next((t["role"] for t in sub_tasks if t["agent_id"] == raw_sub_id), AgentRole.WORKER)

        new_agent_state = self.agent_factory.create_agent(
            user_id=state["user_id"],
            session_id=state["session_id"],
            agent_id=sub_agent_id,
            current_quota=state["quota"],
            parent_depth=state["current_depth"]
        )

        if not new_agent_state:
            return {
                "messages": [{"role": "user", "content": f"[System] Failed to spawn {sub_agent_id}: Quota or Depth limit reached."}],
                "next_steps": state["next_steps"][1:]
            }

        # Handle Recursive Execution if UnitCompiler provided
        if self.unit_compiler:
            logger.info(f"Recursive Execution: Spawning {sub_agent_id} in-thread.")
            child_graph = self.unit_compiler.compile_unit(next_role)
            
            # Prepare child state
            agent_dir = self.agent_factory.workspace.get_agent_dir(state["user_id"], state["session_id"], sub_agent_id)
            child_inbox = agent_dir / "inbox"
            child_outbox = agent_dir / "outbox"
            child_todo = agent_dir / "todo"
            
            child_initial_state = {
                **state,
                "agent_id": sub_agent_id,
                "role": next_role,
                "messages": [SystemMessage(content=prompt or "Process task.")],
                "inbox_path": child_inbox,
                "outbox_path": child_outbox,
                "todo_path": child_todo,
                "current_depth": state["current_depth"] + 1,
                "final_result": None,
                "next_steps": [],
                "node_counts": {} # Reset node counts for child
            }
            
            child_final_state = await child_graph.ainvoke(child_initial_state)
            
            result_val = child_final_state.get("final_result", "Task completed.")
            
            # Format message based on result type
            if isinstance(result_val, dict) and result_val.get("type") == "reference":
                content = f"[System] Sub-agent {sub_agent_id} offloaded result to: {result_val.get('path')}\nSummary: {result_val.get('summary')}"
            elif isinstance(result_val, dict) and result_val.get("type") == "inline":
                content = f"[System] Sub-agent {sub_agent_id} returned: {result_val.get('content')}"
            else:
                content = f"[System] Sub-agent {sub_agent_id} returned: {result_val}"

            return {
                "quota": SessionQuota(agent_count=1),
                "messages": [{"role": "user", "content": content}],
                "next_steps": state["next_steps"][1:]
            }

        # Async Execution via Mailbox
        self.mailbox.send(sub_agent_id, {
            "id": f"task_{state['agent_id']}",
            "sender": state["agent_id"],
            "system_prompt": prompt,
            "payload": {"instructions": state.get("metadata", {}).get("current_task_instructions")},
            "role": next_role
        })
        logger.info(f"Agent {state['agent_id']} sent task to {sub_agent_id} via Mailbox")

        return {
            "quota": SessionQuota(agent_count=1),
            "messages": [{"role": "user", "content": f"[System] Spawned {sub_agent_id} via Mailbox."}],
            "next_steps": state["next_steps"][1:] 
        }

    def _should_continue(self, state: AgentState) -> str:
        # Check for infinite loops
        if state["node_counts"].get("plan", 0) > 100:
            return "abort"
        
        strategy = state.get("metadata", {}).get("strategy")
        if strategy == ExecutionStrategy.DECOMPOSE:
            return "generate_prompt"
        elif strategy == ExecutionStrategy.TOOL_USE:
            return "tools"
        elif strategy == ExecutionStrategy.FINISH:
            return END
        
        # If strategy is missing (parsing failed), retry
        return "plan"

    def _route_after_spawn(self, state: AgentState) -> str:
        """Determines if we should return to planning to observe result."""
        return "plan"

    def build_graph(self, checkpointer: Optional[BaseCheckpointSaver] = None, tool_node: Optional[Any] = None) -> Any:
        workflow = StateGraph(AgentState)
        workflow.add_node("plan", self.planning_node)
        workflow.add_node("generate_prompt", self.generate_prompt_node)
        workflow.add_node("spawn", self.spawning_node)
        workflow.add_node("abort", lambda s: {"messages": [{"role": "user", "content": "[System] ABORTING: Repetition limit reached."}]})
        workflow.add_node("tools", tool_node or (lambda s: {"messages": [{"role": "user", "content": "[System] Tool node stub."}]}))

        workflow.set_entry_point("plan")
        
        workflow.add_conditional_edges(
            "plan", 
            self._should_continue, 
            {
                "generate_prompt": "generate_prompt", 
                "tools": "tools",
                "abort": "abort",
                "plan": "plan",
                END: END
            }
        )
        
        workflow.add_edge("generate_prompt", "spawn")
        workflow.add_conditional_edges(
            "spawn", 
            self._route_after_spawn,
            {
                "generate_prompt": "generate_prompt",
                "plan": "plan"
            }
        )
        workflow.add_edge("tools", "plan")
        workflow.add_edge("abort", END)
        
        return workflow.compile(checkpointer=checkpointer)
