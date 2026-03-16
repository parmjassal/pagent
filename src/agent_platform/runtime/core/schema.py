from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel

class ToolSource(str, Enum):
    COMMUNITY = "community" 
    CORE = "core"
    DYNAMIC = "dynamic"

class ErrorCode(str, Enum):
    GUARDRAIL_BLOCK = "GUARDRAIL_BLOCK"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    TIMEOUT = "TIMEOUT"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    INTERNAL_ERROR = "INTERNAL_ERROR"

class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": False,
            "error": self.message,
            "error_code": self.code.value,
            "details": self.details
        }
