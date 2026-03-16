import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Tuple
from ..orch.state import AgentState

logger = logging.getLogger(__name__)

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
    def generate(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, str]:
        pass

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
        policy_generator: Optional[PolicyGenerator] = None
    ):
        self.lookup = lookup_provider or StableHashLookupProvider()
        self.policy_generator = policy_generator

    def validate_tool_call(
        self, 
        state: AgentState, 
        tool_name: str, 
        tool_args: Dict[str, Any]
    ) -> Tuple[bool, str]:
        user_id = state["user_id"]
        context = {"agent_id": state["agent_id"], "depth": state["current_depth"]}
        action = f"{tool_name}({json.dumps(tool_args)})"

        # 1. Resolve Template (from session workspace)
        session_path = state["inbox_path"].parent.parent.parent
        template_path = session_path / "prompts" / "guardrail_policy.txt"
        system_instruction = "Verify tool call safety."
        if template_path.exists():
            system_instruction = template_path.read_text()

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
            "policy": system_instruction 
        }

        # 2. Check Lookup (Cache)
        cached = self.lookup.get_decision(key_data)
        if cached:
            logger.info(f"Guardrail lookup hit for {action}")
            return cached

        # 3. Invoke Policy Generator (Injected)
        if not self.policy_generator:
            return False, "Guardrail policy generator not configured."

        is_allowed, reason = self.policy_generator.generate(tool_name, tool_args)

        # 4. Store Result
        self.lookup.store_decision(key_data, (is_allowed, reason))
        return is_allowed, reason

def guardrail_tool_wrapper(manager: GuardrailManager):
    def decorator(func: Callable):
        def wrapper(state: AgentState, *args, **kwargs):
            is_allowed, reason = manager.validate_tool_call(state, func.__name__, kwargs)
            if not is_allowed:
                return {"error": reason, "status": "blocked"}
            return func(state, *args, **kwargs)
        return wrapper
    return decorator
