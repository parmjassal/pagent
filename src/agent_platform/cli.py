import typer
import getpass
import asyncio
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from .runtime.bootstrap import start_runtime
from .runtime.core.scheduler import AutonomousScheduler
from .runtime.core.workspace import WorkspaceContext

app = typer.Typer()
console = Console()

async def run_platform(
    task: Optional[str],
    user_id: str,
    session_id: Optional[str],
    openai_base_url: Optional[str],
    model_name: str
):
    # 1. Start Runtime (Bootstrap directories & inject initial task)
    resolved_session_id = start_runtime(
        user_id=user_id, 
        session_id=session_id,
        openai_base_url=openai_base_url,
        model_name=model_name,
        task=task
    )
    
    # 2. Build Rich UI
    tree = Tree(f"[bold cyan]Session: {resolved_session_id}[/bold cyan]")
    infra = tree.add("Infrastructure")
    infra.add(f"User: [green]{user_id}[/green]")
    infra.add(f"Model: [blue]{model_name}[/blue]")
    
    if task:
        infra.add(f"Initial Task: [yellow]{task}[/yellow]")
    
    agents = tree.add("Agent Tree")
    if task:
        agents.add("[bold green]Scheduler active. Processing supervisor...[/bold green]")
    else:
        agents.add("[dim italic]Waiting for task decomposition...[/dim italic]")

    # 3. Start Scheduler
    workspace = WorkspaceContext()
    scheduler = AutonomousScheduler(
        workspace=workspace,
        user_id=user_id,
        session_id=resolved_session_id,
        model_name=model_name,
        openai_base_url=openai_base_url
    )

    console.print(Panel.fit("[bold blue]pagent[/bold blue] - Multi-Agent Runtime", border_style="blue"))
    
    with Live(tree, console=console, refresh_per_second=4):
        # Run the scheduler loop
        try:
            await scheduler.run_forever()
        except asyncio.CancelledError:
            console.print("\n[yellow]Shutdown requested.[/yellow]")

@app.command()
def serve(
    task: Optional[str] = typer.Argument(None, help="The initial task for the agent platform to execute."),
    user_id: Optional[str] = typer.Option(None, help="The ID of the user. Defaults to current process user."),
    session_id: Optional[str] = typer.Option(None, help="The session ID to resume."),
    openai_base_url: Optional[str] = typer.Option(None, envvar="OPENAI_BASE_URL", help="Custom OpenAI endpoint."),
    model_name: str = typer.Option("gpt-4o", envvar="AGENT_MODEL_NAME", help="The LLM model name to use.")
):
    """
    Starts the pagent runtime. 
    If a TASK is provided, it initializes the session with that objective.
    """
    if not user_id:
        user_id = getpass.getuser()
        
    try:
        asyncio.run(run_platform(task, user_id, session_id, openai_base_url, model_name))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app()
