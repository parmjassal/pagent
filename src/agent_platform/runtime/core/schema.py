from enum import Enum

class ToolSource(str, Enum):
    COMMUNITY = "community" 
    CORE = "core"
    DYNAMIC = "dynamic"
