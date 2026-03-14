import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Tuple
from .state import AgentState

logger = logging.getLogger(__name__)

class PolicyLookupProvider(ABC):
    """Interface for looking up existing policy decisions."""
    @abstractmethod
    def get_decision(self, key_data: Dict[str, Any]) -> Optional[Tuple[bool, str]]:
        pass

    @abstractmethod
    def store_decision(self, key_data: Dict[str, Any], decision: Tuple[bool, str]):
        pass

class StableHashLookupProvider(PolicyLookupProvider):
    """Traditional hashing lookup (Fragile but fast)."""
    def __init__(self):
        self.cache: Dict[str, Tuple[bool, str]] = {}

    def _hash(self, data: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get_decision(self, key_data: Dict[str, Any]) -> Optional[Tuple[bool, str]]:
        return self.cache.get(self._hash(key_data))

    def store_decision(self, key_data: Dict[str, Any], decision: Tuple[bool, str]):
        self.cache[self._hash(key_data)] = decision

class SemanticLookupProvider(PolicyLookupProvider):
    """
    Placeholder for LSH or SPLADE-based lookup.
    Allows for 'fuzzy' matching of actions and contexts.
    """
    def get_decision(self, key_data: Dict[str, Any]) -> Optional[Tuple[bool, str]]:
        # TODO: Implement SPLADE explosion on query path
        # TODO: Implement LSH for high-dimensional semantic vector search
        return None 

    def store_decision(self, key_data: Dict[str, Any], decision: Tuple[bool, str]):
        pass

class GuardrailManager:
    """
    Handles context-aware policy validation using a pluggable Lookup Provider.
    """

    def __init__(self, lookup_provider: Optional[PolicyLookupProvider] = None):
        self.lookup = lookup_provider or StableHashLookupProvider()

    def validate_tool_call(
        self, 
        state: AgentState, 
        tool_name: str, 
        tool_args: Dict[str, Any]
    ) -> Tuple[bool, str]:
        user_id = state["user_id"]
        context = {"agent_id": state["agent_id"], "depth": state["current_depth"]}
        action = f"{tool_name}({json.dumps(tool_args)})"
        
        history = ""
        if state.get("messages"):
            history = " ".join([m["content"] for m in state["messages"][-2:]])

        key_data = {
            "user_id": user_id,
            "context": context,
            "action": action,
            "history": history
        }

        # 1. Check Lookup (Hash or Semantic)
        cached = self.lookup.get_decision(key_data)
        if cached:
            logger.info(f"Guardrail lookup hit for {action}")
            return cached

        # 2. Simulated Policy Check
        is_allowed, reason = self._simulated_policy_engine(tool_name, tool_args)

        # 3. Store Result
        self.lookup.store_decision(key_data, (is_allowed, reason))
        return is_allowed, reason

    def _simulated_policy_engine(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, str]:
        if "delete" in tool_name.lower() or "rm" in tool_name.lower():
            return False, f"Policy violation: Destructive tool '{tool_name}' is prohibited."
        return True, "Allowed"

def guardrail_tool_wrapper(manager: GuardrailManager):
    def decorator(func: Callable):
        def wrapper(state: AgentState, *args, **kwargs):
            is_allowed, reason = manager.validate_tool_call(state, func.__name__, kwargs)
            if not is_allowed:
                return {"error": reason, "status": "blocked"}
            return func(state, *args, **kwargs)
        return wrapper
    return decorator
