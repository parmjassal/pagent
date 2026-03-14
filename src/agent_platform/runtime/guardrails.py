import hashlib
import json
import logging
from typing import Dict, Any, Optional, Callable, Tuple
from .state import AgentState

logger = logging.getLogger(__name__)

class GuardrailManager:
    """
    Handles context-aware policy generation and validation for tool execution.
    Includes caching to minimize latency of LLM/SMT-based checks.
    """

    def __init__(self):
        # In-memory cache for policy results: {(hash): (is_allowed, reason)}
        self._policy_cache: Dict[str, Tuple[bool, str]] = {}

    def _generate_cache_key(
        self, 
        user_id: str, 
        context: Dict[str, Any], 
        action: str, 
        history_summary: str
    ) -> str:
        """Creates a unique hash for the current execution context."""
        payload = {
            "user_id": user_id,
            "context": context,
            "action": action,
            "history": history_summary
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def validate_tool_call(
        self, 
        state: AgentState, 
        tool_name: str, 
        tool_args: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validates a tool call against guardrail policies.
        Returns (is_allowed, reason).
        """
        user_id = state["user_id"]
        # Context includes agent_id and current_depth
        context = {"agent_id": state["agent_id"], "depth": state["current_depth"]}
        action = f"{tool_name}({json.dumps(tool_args)})"
        
        # Simple history summary from last 2 messages
        history = ""
        if state.get("messages"):
            history = " ".join([m["content"] for m in state["messages"][-2:]])

        cache_key = self._generate_cache_key(user_id, context, action, history)

        # 1. Check Cache
        if cache_key in self._policy_cache:
            logger.info(f"Guardrail cache hit for {action}")
            return self._policy_cache[cache_key]

        # 2. Simulated Policy Check (LLM/SMT Logic)
        # In production, this would call the GuardrailPolicyGenerator (LLM or SMT)
        is_allowed, reason = self._simulated_policy_engine(tool_name, tool_args)

        # 3. Store in Cache
        self._policy_cache[cache_key] = (is_allowed, reason)
        return is_allowed, reason

    def _simulated_policy_engine(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, str]:
        """Placeholder for actual LLM/SMT policy validation."""
        # Example rule: No deletion tools allowed for now
        if "delete" in tool_name.lower() or "rm" in tool_name.lower():
            return False, f"Policy violation: Destructive tool '{tool_name}' is prohibited."
        
        return True, "Allowed"

def guardrail_tool_wrapper(manager: GuardrailManager):
    """Decorator/Wrapper for tool execution."""
    def decorator(func: Callable):
        def wrapper(state: AgentState, *args, **kwargs):
            tool_name = func.__name__
            # Assuming tool_args are passed as kwargs
            is_allowed, reason = manager.validate_tool_call(state, tool_name, kwargs)
            
            if not is_allowed:
                logger.warning(f"Guardrail BLOCKED tool execution: {reason}")
                return {"error": reason, "status": "blocked"}
            
            return func(state, *args, **kwargs)
        return wrapper
    return decorator
