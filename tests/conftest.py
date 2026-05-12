from pathlib import Path

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[1]
