import typer
import getpass
from typing import Optional
from .runtime.bootstrap import start_runtime

app = typer.Typer()

@app.command()
def serve(
    user_id: Optional[str] = typer.Option(None, help="The ID of the user. Defaults to the current system user."),
    session_id: Optional[str] = typer.Option(None, help="The session ID to resume. If not provided, a new one is created."),
    openai_base_url: Optional[str] = typer.Option(None, envvar="OPENAI_BASE_URL", help="Custom base URL for OpenAI API (useful for proxies/local models).")
):
    """
    Starts the agent platform runtime for a specific user and session context.
    """
    if not user_id:
        user_id = getpass.getuser()
        
    resolved_session_id = start_runtime(
        user_id=user_id, 
        session_id=session_id,
        openai_base_url=openai_base_url
    )
    
    print(f"Platform running for user: {user_id}")
    if session_id:
        print(f"Connected to existing session: {resolved_session_id}")
    else:
        print(f"Created new session: {resolved_session_id}")

if __name__ == "__main__":
    app()
