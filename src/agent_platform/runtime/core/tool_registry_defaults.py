from .schema import ToolSource

# The "Important 19" - Static Tool Manifest
# This list is loaded by the ToolRegistry at compile time.

STATIC_TOOL_REGISTRY = [
    # --- Core Platform Tools ---
    {"name": "ls", "source": ToolSource.CORE, "summary": "List files and directories in a given path."},
    {"name": "read_file", "source": ToolSource.CORE, "summary": "Read content from a specific file path."},
    {"name": "write_file", "source": ToolSource.CORE, "summary": "Writes content to a file within the session workspace. Creates parent directories if needed."},
    {"name": "grep_search", "source": ToolSource.CORE, "summary": "Find regex patterns within files."},
    {"name": "semantic_search", "source": ToolSource.CORE, "summary": "Perform semantic queries over the indexed workspace."},
    {"name": "build_index", "source": ToolSource.CORE, "summary": "Index a local directory for semantic retrieval."},
    {"name": "add_task", "source": ToolSource.CORE, "summary": "Create a new scoped task in the session TODO."},
    {"name": "list_tasks", "source": ToolSource.CORE, "summary": "Retrieve and filter tasks from the session TODO."},
    {"name": "update_task", "source": ToolSource.CORE, "summary": "Mark a task as completed or failed."},
    {"name": "extract_facts", "source": ToolSource.CORE, "summary": "Generate a Markdown fact sheet from a file chunk."},
    {"name": "list_knowledge", "source": ToolSource.CORE, "summary": "Show all extracted facts in the current session."},
    
    # --- Meta-Tool for Dynamic Capability ---
    {"name": "write_tool", "source": ToolSource.CORE, "summary": "Generate a new Python tool for a specific task and add it to the session skills."},

    # --- Extended Community Tools ---
    {"name": "web_search", "source": ToolSource.COMMUNITY, "summary": "Search the internet for real-time information (e.g., Tavily)."},
    {"name": "wikipedia", "source": ToolSource.COMMUNITY, "summary": "Fetch summaries and facts from Wikipedia."},
    {"name": "shell_exec", "source": ToolSource.COMMUNITY, "summary": "Execute bash commands (Always Sandboxed)."},
    {"name": "python_repl", "source": ToolSource.COMMUNITY, "summary": "Run Python code for complex math/logic (Always Sandboxed)."},
    {"name": "http_get", "source": ToolSource.COMMUNITY, "summary": "Fetch raw content from a URL."},
    {"name": "git_clone", "source": ToolSource.COMMUNITY, "summary": "Clone a git repository into the user workspace."},
    {"name": "diff_check", "source": ToolSource.COMMUNITY, "summary": "Compare two files or strings for differences."},
    {"name": "json_extract", "source": ToolSource.COMMUNITY, "summary": "Extract specific fields from large/complex JSON data."},
]
