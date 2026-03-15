from typing import List, Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from ..orch.state import AgentState, AgentRole
from ..orch.quota import SessionQuota
from ..core.agent_factory import AgentFactory
from ..core.todo import TODOManager, ScopedTask
from .generator import SystemGeneratorAgent, TaskType
from ..orch.logic import LoopMonitor
from ..core.http_client import get_platform_http_client
from ..orch.models import DecompositionResult, PlanningResult, ExecutionStrategy
from ..core.mailbox import Mailbox

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """The primary orchestrator that plans, decomposes tasks or executes tools."""

    def __init__(
        self, 
        agent_factory: AgentFactory,
        mailbox: Mailbox,
        generator: SystemGeneratorAgent,
        model_name: str = "gpt-4o", 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,
        llm: Optional[Any] = None,
        unit_compiler: Optional[Any] = None 
    ):
        self.agent_factory = agent_factory
        self.mailbox = mailbox
        self.generator = generator
        self.unit_compiler = unit_compiler
        
        self.parser = JsonOutputParser(pydantic_object=PlanningResult)
        
        if llm:
            self.llm = llm
        else:
            http_client = get_platform_http_client()
            self.base_llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                http_client=http_client,
                temperature=0 
            )
            self.llm = self.base_llm | self.parser

    def _should_continue(self, state: AgentState) -> str:
        """Determines the next path based on the chosen strategy."""
        strategy = state.get("metadata", {}).get("strategy")
        
        node_threshold = 3 if state["role"] == AgentRole.SUPERVISOR else 10
        if LoopMonitor.check_node_loop(state, "plan", threshold=node_threshold):
            return "abort"
        if LoopMonitor.check_content_loop(state, window=3):
            return "abort"

        if strategy == ExecutionStrategy.DECOMPOSE:
            return "generate_prompt"
        elif strategy == ExecutionStrategy.TOOL_USE:
            return "tools"
        elif strategy == ExecutionStrategy.FINISH:
            return END
        
        return END

    async def planning_node(self, state: AgentState) -> AgentState:
        """Invokes the LLM to decide on an execution strategy."""
        template_path = state["inbox_path"].parent.parent.parent / "prompts" / "supervisor_decompose.txt"
        system_instruction = "You are a task planning supervisor."
        if template_path.exists():
            system_instruction = template_path.read_text()

        format_instructions = self.parser.get_format_instructions()
        full_instruction = f"{system_instruction}\n\n{format_instructions}"

        prompt = [
            SystemMessage(content=full_instruction),
            *state["messages"]
        ]
        
        raw_result = await self.llm.ainvoke(prompt)
        result = PlanningResult.model_validate(raw_result)
        
        metadata_update = {
            "strategy": result.strategy,
            "thought_process": result.thought_process
        }

        if result.strategy == ExecutionStrategy.DECOMPOSE and result.sub_tasks:
            todo_mgr = TODOManager(state["todo_path"].parent)
            next_steps = []
            for task_def in result.sub_tasks:
                todo_mgr.add_task(ScopedTask(
                    title=f"Task for {task_def.agent_id}",
                    description=task_def.instructions,
                    assigned_to=task_def.agent_id,
                    metadata={"role": task_def.role}
                ))
                next_steps.append(task_def.agent_id)
            
            metadata_update["next_agent_role"] = result.sub_tasks[0].role
            metadata_update["current_task_instructions"] = result.sub_tasks[0].instructions
            
            return {
                "role": state["role"],
                "messages": [{"role": "assistant", "content": result.thought_process}],
                "next_steps": next_steps,
                "metadata": metadata_update,
                "node_counts": {"plan": 1}
            }
        
        elif result.strategy == ExecutionStrategy.TOOL_USE and result.tool_call:
            metadata_update["next_tool_call"] = result.tool_call.model_dump()
            return {
                "role": state["role"],
                "messages": [{"role": "assistant", "content": result.thought_process}],
                "metadata": metadata_update,
                "node_counts": {"plan": 1}
            }

        return {
            "role": state["role"],
            "messages": [{"role": "assistant", "content": result.thought_process}],
            "metadata": metadata_update,
            "node_counts": {"plan": 1}
        }

    async def generate_prompt_node(self, state: AgentState) -> Dict[str, Any]:
        return await self.generator.generate_node(state, task_type=TaskType.PROMPT)

    async def spawning_node(self, state: AgentState) -> AgentState:
        if not state["next_steps"]:
            return state

        sub_agent_id = state["next_steps"][0]
        prompt = state.get("generated_output")
        next_role = state.get("metadata", {}).get("next_agent_role", AgentRole.WORKER)

        new_agent_state = self.agent_factory.create_agent(
            user_id=state["user_id"],
            session_id=state["session_id"],
            agent_id=sub_agent_id,
            current_quota=state["quota"],
            parent_depth=state["current_depth"],
            generated_output=prompt,
            role=next_role 
        )

        if not new_agent_state:
            return {"messages": [{"role": "system", "content": f"Failed to spawn {sub_agent_id}"}]}
        
        if self.unit_compiler:
            logger.info(f"Supervisor {state['agent_id']} AWAITING subgraph: {sub_agent_id}")
            sub_graph = self.unit_compiler.compile_unit(role=next_role)
            sub_input = {**state, **new_agent_state}
            sub_result_state = await sub_graph.ainvoke(sub_input)
            final_res = sub_result_state.get("final_result", {"content": "Sub-task finished."})
            
            return {
                "quota": sub_result_state["quota"],
                "messages": [{"role": "system", "content": f"Sub-agent {sub_agent_id} returned: {final_res}"}],
                "next_steps": state["next_steps"][1:]
            }
        
        self.mailbox.send(sub_agent_id, {
            "id": f"task_{state['agent_id']}",
            "sender": state["agent_id"],
            "system_prompt": prompt,
            "payload": {"instructions": state.get("metadata", {}).get("current_task_instructions")},
            "role": next_role
        })

        return {
            "quota": SessionQuota(agent_count=1),
            "messages": [{"role": "system", "content": f"Spawned {sub_agent_id} via Mailbox."}],
            "next_steps": state["next_steps"][1:] 
        }

    def abort_node(self, state: AgentState) -> Dict[str, Any]:
        return {"messages": [{"role": "system", "content": f"ABORTING: Loop detected for {state['role']} agent."}]}

    def dummy_tool_node(self, state: AgentState) -> Dict[str, Any]:
        return {"messages": [{"role": "system", "content": "Tool execution simulation."}]}

    def build_graph(self, checkpointer: Optional[BaseCheckpointSaver] = None, tool_node: Optional[Any] = None) -> Any:
        workflow = StateGraph(AgentState)
        workflow.add_node("plan", self.planning_node)
        workflow.add_node("generate_prompt", self.generate_prompt_node)
        workflow.add_node("spawn", self.spawning_node)
        workflow.add_node("abort", self.abort_node)
        
        # Always add a 'tools' node, using a dummy if none provided
        workflow.add_node("tools", tool_node or self.dummy_tool_node)

        workflow.set_entry_point("plan")
        
        workflow.add_conditional_edges(
            "plan", 
            self._should_continue, 
            {
                "generate_prompt": "generate_prompt", 
                "tools": "tools",
                "abort": "abort",
                END: END
            }
        )
        
        workflow.add_edge("generate_prompt", "spawn")
        workflow.add_edge("spawn", "plan")
        workflow.add_edge("tools", "plan")
        workflow.add_edge("abort", END)
        
        return workflow.compile(checkpointer=checkpointer)
