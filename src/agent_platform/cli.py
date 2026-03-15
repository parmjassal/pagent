import typer
import getpass
import time
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from .runtime.bootstrap import start_runtime

app = typer.Typer()
console = Console()

@app.command()
def serve(
    user_id: Optional[str] = typer.Option(None, help="The ID of the user. Defaults to current process user."),
    session_id: Optional[str] = typer.Option(None, help="The session ID to resume."),
    openai_base_url: Optional[str] = typer.Option(None, envvar="OPENAI_BASE_URL", help="Custom OpenAI endpoint."),
    model_name: str = typer.Option("gpt-4o", envvar="AGENT_MODEL_NAME", help="The LLM model name to use.")
):
    """
    Starts the pagent runtime with a real-time visualization of the orchestration tree.
    """
    if not user_id:
        user_id = getpass.getuser()
        
    # 1. Start Runtime
    resolved_session_id = start_runtime(
        user_id=user_id, 
        session_id=session_id,
        openai_base_url=openai_base_url,
        model_name=model_name
    )
    
    # 2. Build Rich UI
    tree = Tree(f"[bold cyan]Session: {resolved_session_id}[/bold cyan]")
    infra = tree.add("Infrastructure")
    infra.add(f"User: [green]{user_id}[/green]")
    infra.add(f"Model: [blue]{model_name}[/blue]")
    infra.add("Mailbox: [blue]Listening...[/blue]")
    
    agents = tree.add("Agent Tree")
    agents.add("[dim italic]Waiting for task decomposition...[/dim italic]")

    # 3. Live Display
    console.print(Panel.fit("[bold blue]pagent[/bold blue] - Multi-Agent Runtime", border_style="blue"))
    
    with Live(tree, console=console, refresh_per_second=4):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutdown requested.[/yellow]")

if __name__ == "__main__":
    app()
