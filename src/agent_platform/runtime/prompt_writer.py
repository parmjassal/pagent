from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .state import AgentState
from .workspace import WorkspaceContext

class DynamicPromptWriter:
    """
    System Agent responsible for generating context-aware system prompts 
    for newly spawned sub-agents.
    """

    def __init__(self, llm: ChatOpenAI, workspace: WorkspaceContext):
        self.llm = llm
        self.workspace = workspace

    def generate_prompt_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node to generate a tailored system prompt for a sub-agent."""
        
        # 1. Identify the task from the last decomposition result
        if not state["messages"]:
            return {"generated_prompt": "You are a specialized agent."}
            
        parent_task_desc = state["messages"][-1]["content"]
        sub_agent_id = state["next_steps"][0] if state["next_steps"] else "sub_agent"

        # 2. Resolve session prompt templates
        # (This is where the 'Copy Rule' shines - we look at session-local prompts)
        session_prompt_dir = self.workspace.get_session_dir(state["user_id"], state["session_id"]) / "prompts"
        base_template = "You are a specialized AI assistant."
        
        template_file = session_prompt_dir / "agent_base.txt"
        if template_file.exists():
            base_template = template_file.read_text()

        # 3. Use LLM to tailor the prompt
        # In a real scenario, this would be a more complex LangChain prompt template
        human_instruction = f"Generate a system prompt for a sub-agent named {sub_agent_id} who must: {parent_task_desc}. Base it on: {base_template}"
        
        # response = self.llm.invoke([SystemMessage(content="You are a Meta-Prompt Engineer."), HumanMessage(content=human_instruction)])
        # For testing/demo without active LLM, we'll simulate the output:
        generated = f"SYSTEM PROMPT for {sub_agent_id}: {parent_task_desc}. {base_template}"

        return {
            "messages": [{"role": "system", "content": f"Prompt Writer: Generated prompt for {sub_agent_id}"}],
            "generated_prompt": generated
        }
