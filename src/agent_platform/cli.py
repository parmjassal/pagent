import typer
import getpass
import asyncio
import os
from typing import Optional, Dict
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from .runtime.bootstrap import start_runtime
from .runtime.core.scheduler import AutonomousScheduler
from .runtime.core.workspace import WorkspaceContext

app = typer.Typer()
console = Console()

def build_dynamic_tree(session_id: str, user_id: str, model_name: str, task: Optional[str], session_path: Path) -> Tree:
    """Recursively builds a tree representing the current session and agents."""
    tree = Tree(f"[bold cyan]Session: {session_id}[/bold cyan]")
    
    # 1. Infrastructure Info
    infra = tree.add("Infrastructure")
    infra.add(f"User: [green]{user_id}[/green]")
    infra.add(f"Model: [blue]{model_name}[/blue]")
    if task:
        infra.add(f"Task: [dim]{task[:50]}...[/dim]")

    # 2. Agent Hierarchy (Discovered from Filesystem)
    agents_root = session_path / "agents"
    agents_tree = tree.add("[bold magenta]Agent Tree[/bold magenta]")
    
    if not agents_root.exists():
        agents_tree.add("[dim]Initializing...[/dim]")
        return tree

    # Map to keep track of added nodes to prevent duplicates in recursive walk
    nodes = {agents_root: agents_tree}

    # Walk the directory tree to find agents
    # We look for directories that contain a 'todo' or 'inbox' to identify them as agents
    for root, dirs, files in os.walk(agents_root):
        root_path = Path(root)
        
        # If this is an agent directory
        if (root_path / "todo").exists() or (root_path / "inbox").exists():
            parent_path = root_path.parent
            parent_node = nodes.get(parent_path, agents_tree)
            
            # Extract status from metadata if possible (simplified for now)
            status = "[green]Active[/green]"
            if (root_path / "outbox").exists() and any((root_path / "outbox").glob("*.json")):
                status = "[blue]Responded[/blue]"
            
            agent_node = parent_node.add(f"Agent: [bold yellow]{root_path.name}[/bold yellow] ({status})")
            nodes[root_path] = agent_node

    return tree

async def run_platform(
    task: Optional[str],
    user_id: str,
    session_id: Optional[str],
    openai_base_url: Optional[str],
    model_name: str
):
    # 1. Start Runtime
    resolved_session_id = start_runtime(
        user_id=user_id, 
        session_id=session_id,
        openai_base_url=openai_base_url,
        model_name=model_name,
        task=task
    )
    
    workspace = WorkspaceContext()
    session_path = workspace.get_session_dir(user_id, resolved_session_id)
    
    # 2. Start Scheduler in background
    scheduler = AutonomousScheduler(
        workspace=workspace,
        user_id=user_id,
        session_id=resolved_session_id,
        model_name=model_name,
        openai_base_url=openai_base_url
    )

    scheduler_task = asyncio.create_task(scheduler.run_forever())

    # 3. Dynamic UI Loop
    console.print(Panel.fit("[bold blue]pagent[/bold blue] - Multi-Agent Runtime", border_style="blue"))
    
    with Live(refresh_per_second=2) as live:
        try:
            while not scheduler_task.done():
                # Rebuild tree from current filesystem state
                current_tree = build_dynamic_tree(
                    resolved_session_id, user_id, model_name, task, session_path
                )
                live.update(current_tree)
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            if not scheduler_task.done():
                scheduler_task.cancel()

@app.command()
def serve(
    task: Optional[str] = typer.Argument(None, help="The initial task for the agent platform to execute."),
    user_id: Optional[str] = typer.Option(None, help="The ID of the user. Defaults to current process user."),
    session_id: Optional[str] = typer.Option(None, help="The session ID to resume."),
    openai_base_url: Optional[str] = typer.Option(None, envvar="OPENAI_BASE_URL", help="Custom OpenAI endpoint."),
    model_name: str = typer.Option("gpt-4o", envvar="AGENT_MODEL_NAME", help="The LLM model name to use.")
):
    """Starts the pagent runtime with dynamic hierarchical visualization."""
    if not user_id:
        user_id = getpass.getuser()
        
    try:
        asyncio.run(run_platform(task, user_id, session_id, openai_base_url, model_name))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app()
