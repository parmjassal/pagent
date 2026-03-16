from .schema import ToolSource

# The "Important 19" - Static Tool Manifest
# This list is loaded by the ToolRegistry at compile time.

STATIC_TOOL_REGISTRY = [
    # --- Core Platform Tools ---
    {"name": "ls", "source": ToolSource.CORE, "summary": "List files and directories.", "parameters": ["path"]},
    {"name": "read_file", "source": ToolSource.CORE, "summary": "Read content from a file.", "parameters": ["file_path"]},
    {"name": "write_file", "source": ToolSource.CORE, "summary": "Write content to a file.", "parameters": ["file_path", "content"]},
    {"name": "grep_search", "source": ToolSource.CORE, "summary": "Find regex patterns within files.", "parameters": ["pattern", "path"]},
    {"name": "semantic_search", "source": ToolSource.CORE, "summary": "Perform semantic queries.", "parameters": ["query"]},
    {"name": "build_index", "source": ToolSource.CORE, "summary": "Index a directory for semantic retrieval.", "parameters": ["path"]},
    {"name": "add_task", "source": ToolSource.CORE, "summary": "Create a new scoped task.", "parameters": ["title", "description"]},
    {"name": "list_tasks", "source": ToolSource.CORE, "summary": "Retrieve and filter tasks.", "parameters": ["status"]},
    {"name": "update_task", "source": ToolSource.CORE, "summary": "Update task status.", "parameters": ["task_id", "status"]},
    {"name": "extract_facts", "source": ToolSource.CORE, "summary": "Generate fact sheet from file.", "parameters": ["file_path", "fact_id"]},
    {"name": "list_knowledge", "source": ToolSource.CORE, "summary": "Show all extracted facts.", "parameters": []},
    
    # --- Meta-Tool for Dynamic Capability ---
    {"name": "write_tool", "source": ToolSource.CORE, "summary": "Generate a new Python tool.", "parameters": ["name", "code", "summary"]},

    # --- Extended Community Tools ---
    {"name": "web_search", "source": ToolSource.COMMUNITY, "summary": "Search the internet.", "parameters": ["query"]},
    {"name": "wikipedia", "source": ToolSource.COMMUNITY, "summary": "Fetch Wikipedia summaries.", "parameters": ["query"]},
    {"name": "shell_exec", "source": ToolSource.COMMUNITY, "summary": "Execute bash commands.", "parameters": ["command"]},
    {"name": "python_repl", "source": ToolSource.COMMUNITY, "summary": "Run Python code.", "parameters": ["code"]},
    {"name": "http_get", "source": ToolSource.COMMUNITY, "summary": "Fetch raw content from a URL.", "parameters": ["url"]},
    {"name": "git_clone", "source": ToolSource.COMMUNITY, "summary": "Clone a git repository.", "parameters": ["url"]},
    {"name": "diff_check", "source": ToolSource.COMMUNITY, "summary": "Compare two files.", "parameters": ["file_a", "file_b"]},
    {"name": "json_extract", "source": ToolSource.COMMUNITY, "summary": "Extract fields from JSON.", "parameters": ["json_str", "path"]},
]
