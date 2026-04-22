import os
import wandb
import streamlit as st


def login(api_key: str) -> bool:
    """Attempt W&B login. Returns True on success."""
    os.environ["WANDB_API_KEY"] = api_key.strip()
    try:
        wandb.login(key=api_key.strip(), relogin=True)
        return True
    except Exception as e:
        st.warning(f"W&B login failed: {e}. Running locally.")
        return False


def log_run(project: str, run_name: str, config: dict, metrics: dict):
    """Log a single experiment run to W&B."""
    try:
        run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            reinit=True
        )
        wandb.log(metrics)
        wandb.finish()
    except Exception:
        pass


def finish():
    """Safely finish any active W&B run."""
    try:
        wandb.finish()
    except Exception:
        pass
