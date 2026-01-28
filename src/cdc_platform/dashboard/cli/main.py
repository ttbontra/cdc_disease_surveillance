from __future__ import annotations
import typer
from .commands.ingest import ingest_cmd
from .commands.build_features import features_cmd
from .commands.train_models import train_risk_cmd
from .commands.serve import serve_cmd

app = typer.Typer(add_completion=False)

app.command("ingest")(ingest_cmd)
app.command("build-features")(features_cmd)
app.command("train-risk")(train_risk_cmd)
app.command("serve")(serve_cmd)

def main():
    app()

if __name__ == "__main__":
    main()
