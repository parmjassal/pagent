import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from ..orch.state import AgentState
from ..core.http_client import get_platform_http_client
from ..core.context_store import ContextStore

logger = logging.getLogger(__name__)

class PolicyDecision(BaseModel):
    is_allowed: bool = Field(description="Whether the tool call is permitted")
    reason: str = Field(description="The justification for the decision")

class PolicyLookupProvider(ABC):
    """Interface for looking up existing policy decisions."""
    @abstractmethod
    def get_decision(self, key_data: Dict[str, Any]) -> Optional[Tuple[bool, str]]:
        pass

    @abstractmethod
    def store_decision(self, key_data: Dict[str, Any], decision: Tuple[bool, str]):
        pass

class PolicyGenerator(ABC):
    """Interface for dynamically generating a policy decision (LLM or SMT)."""
    @abstractmethod
    async def generate(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any], 
        history: str, 
        policy_text: str,
        visible_context: Optional[str] = None
    ) -> Tuple[bool, str]:
        pass

class LLMPolicyGenerator(PolicyGenerator):
    """LLM-based policy generator."""
    def __init__(self, model_name: str = "gpt-4o", base_url: Optional[str] = None):
        http_client = get_platform_http_client()
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_base=base_url,
            http_client=http_client,
            temperature=0
        )
        self.parser = JsonOutputParser(pydantic_object=PolicyDecision)

    async def generate(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any], 
        history: str, 
        policy_text: str,
        visible_context: Optional[str] = None
    ) -> Tuple[bool, str]:
        format_instructions = self.parser.get_format_instructions()
        
        system_msg = SystemMessage(content=f"{policy_text}\n\n{format_instructions}")
        human_data = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "recent_history": history,
            "global_context": visible_context or "No additional global context available."
        }
        human_msg = HumanMessage(content=json.dumps(human_data))

        try:
            response = await self.llm.ainvoke([system_msg, human_msg])
            
            # Robust Parsing
            content = response.content if hasattr(response, "content") else str(response)
            parsed = self.parser.parse(content)
            decision = PolicyDecision.model_validate(parsed)
            return decision.is_allowed, decision.reason
        except Exception as e:
            logger.error(f"Guardrail LLM failure: {e}")
            # Fail closed for security
            return False, f"Guardrail internal error: {e}"

class StableHashLookupProvider(PolicyLookupProvider):
    def __init__(self):
        self.cache: Dict[str, Tuple[bool, str]] = {}

    def _hash(self, data: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get_decision(self, key_data: Dict[str, Any]) -> Optional[Tuple[bool, str]]:
        return self.cache.get(self._hash(key_data))

    def store_decision(self, key_data: Dict[str, Any], decision: Tuple[bool, str]):
        self.cache[self._hash(key_data)] = decision

class GuardrailManager:
    """
    Handles context-aware policy validation.
    """

    def __init__(
        self, 
        lookup_provider: Optional[PolicyLookupProvider] = None,
        policy_generator: Optional[PolicyGenerator] = None,
        context_store: Optional[ContextStore] = None
    ):
        self.lookup = lookup_provider or StableHashLookupProvider()
        self.policy_generator = policy_generator
        self.context_store = context_store

    async def validate_tool_call(
        self, 
        state: AgentState, 
        tool_name: str, 
        tool_args: Dict[str, Any]
    ) -> Tuple[bool, str]:
        user_id = state["user_id"]
        agent_id = state["agent_id"]
        context = {"agent_id": agent_id, "depth": state["current_depth"]}
        action = f"{tool_name}({json.dumps(tool_args)})"

        # 1. Resolve Template (from session workspace)
        # Check hierarchy for policy file
        session_path = state["inbox_path"].parent.parent.parent
        template_path = session_path / "prompts" / "guardrail_policy.txt"
        
        system_instruction = "Verify if the following tool call is safe and follows system guidelines."
        if template_path.exists():
            system_instruction = template_path.read_text()

        # 2. Fetch Visible Global Context (Facts)
        visible_context = None
        if self.context_store:
            facts = self.context_store.list_facts(agent_id)
            if facts:
                fact_contents = []
                for fid in facts:
                    content = self.context_store.read_fact(agent_id, fid)
                    if content:
                        fact_contents.append(f"### Fact: {fid}\n{content}")
                visible_context = "\n\n".join(fact_contents)

        history = ""
        if state.get("messages"):
            history_content = []
            for m in state["messages"][-3:]:
                if hasattr(m, "content"):
                    history_content.append(str(m.content))
                elif isinstance(m, dict):
                    history_content.append(str(m.get("content", "")))
                else:
                    history_content.append(str(m))
            history = " ".join(history_content)

        key_data = {
            "user_id": user_id,
            "context": context,
            "action": action,
            "history": history,
            "policy": system_instruction,
            "visible_context": visible_context
        }

        # 3. Check Lookup (Cache)
        cached = self.lookup.get_decision(key_data)
        if cached:
            logger.info(f"Guardrail lookup hit for {action}")
            return cached

        # 4. Invoke Policy Generator (Injected)
        if not self.policy_generator:
            # If no generator, we block by default to be safe
            return False, "Guardrail policy generator not configured."

        is_allowed, reason = await self.policy_generator.generate(
            tool_name, 
            tool_args, 
            history, 
            system_instruction,
            visible_context=visible_context
        )

        # 5. Store Result
        self.lookup.store_decision(key_data, (is_allowed, reason))
        return is_allowed, reason

def guardrail_tool_wrapper(manager: GuardrailManager):
    def decorator(func: Callable):
        async def wrapper(state: AgentState, *args, **kwargs):
            is_allowed, reason = await manager.validate_tool_call(state, func.__name__, kwargs)
            if not is_allowed:
                return {"error": reason, "status": "blocked"}
            return func(state, *args, **kwargs)
        return wrapper
    return decorator
