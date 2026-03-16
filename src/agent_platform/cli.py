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

import json

def build_dynamic_tree(session_id: str, user_id: str, model_name: str, task: Optional[str], session_path: Path) -> Tree:
    """Recursively builds a tree representing the current session, agents, and their tasks."""
    tree = Tree(f"🌳 [bold cyan]Session: {session_id}[/bold cyan]")
    
    # 1. Infrastructure Info
    infra = tree.add("🔌 [bold]Infrastructure[/bold]")
    infra.add(f"👤 User: [green]{user_id}[/green]")
    infra.add(f"🧠 Model: [blue]{model_name}[/blue]")
    if task:
        infra.add(f"🎯 Initial Task: [dim]{task[:70]}...[/dim]")

    # 2. Agent Hierarchy
    agents_root = session_path / "agents"
    agents_tree = tree.add("🤖 [bold magenta]Agent & Task Tree[/bold magenta]")
    
    if not agents_root.exists() or not any(agents_root.iterdir()):
        agents_tree.add("[dim]Initializing or no agents created yet...[/dim]")
        return tree

    # --- Data Collection Pass ---
    agents_data = {}
    child_to_parent_map = {}

    agent_dirs = [d for d in agents_root.iterdir() if d.is_dir()]
    for agent_dir in agent_dirs:
        agent_name = agent_dir.name
        agents_data[agent_name] = {'path': agent_dir, 'todos': []}
        
        todo_dir = agent_dir / "todo"
        if todo_dir.exists():
            for todo_file in sorted(todo_dir.glob("*.json")):
                try:
                    with open(todo_file, 'r') as f:
                        todo_data = json.load(f)
                        agents_data[agent_name]['todos'].append(todo_data)
                        if todo_data.get('assigned_to'):
                            child_to_parent_map[todo_data['assigned_to']] = agent_name
                except (json.JSONDecodeError, IOError):
                    continue # Ignore corrupted or unreadable files

    # --- Tree Building Pass ---
    def get_status_style(status: Optional[str]) -> str:
        status_str = f"({status})" if status else ""
        if status == 'completed':
            return f"[green]{status_str}[/green]"
        if status in ['in_progress', 'running']:
            return f"[yellow]{status_str}[/yellow]"
        if status == 'pending':
            return f"[dim]{status_str}[/dim]"
        return f"[dim]{status_str}[/dim]"

    def _build_subtree(parent_node: Tree, agent_name: str):
        if agent_name not in agents_data:
            parent_node.add(f"⚠️ [bold red]Error:[/bold red] Agent '{agent_name}' defined but not found.")
            return

        agent_info = agents_data[agent_name]
        for todo in agent_info['todos']:
            status_style = get_status_style(todo.get('status'))
            description = todo.get('description', 'No description').replace('[', '(').replace(']', ')')
            
            assigned_agent = todo.get('assigned_to')
            if assigned_agent:
                # This todo represents spawning a sub-agent
                sub_agent_node = parent_node.add(f"🤖 [bold yellow]Agent: {assigned_agent}[/bold yellow] - Goal: {description} {status_style}")
                _build_subtree(sub_agent_node, assigned_agent)
            else:
                # This is a regular task for the current agent
                parent_node.add(f"📝 Task: {description} {status_style}")

    # Find the root agent(s) (those not in the child_map)
    root_agents = [name for name in agents_data if name not in child_to_parent_map]
    
    if not root_agents and agents_data:
         # Handle cases with loops or if all agents are children
        root_agents = list(agents_data.keys())


    for root_agent_name in root_agents:
        root_node = agents_tree.add(f"👑 [bold yellow]Root Agent: {root_agent_name}[/bold yellow]")
        _build_subtree(root_node, root_agent_name)
    
    if not agents_data:
        agents_tree.add("[dim]No agents found.[/dim]")

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
    
    with Live(refresh_per_second=2, screen=True) as live:
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
