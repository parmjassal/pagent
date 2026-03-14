import typer
from .runtime.bootstrap import start_runtime

app = typer.Typer()

@app.command()
def serve():
    start_runtime()

if __name__ == "__main__":
    app()
