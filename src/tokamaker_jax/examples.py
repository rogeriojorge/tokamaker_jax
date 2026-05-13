"""Packaged example helpers."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

EXAMPLE_FILES = {
    "fixed-boundary": "fixed_boundary.toml",
}


def available_examples() -> tuple[str, ...]:
    """Return packaged example names."""

    return tuple(sorted(EXAMPLE_FILES))


def example_filename(name: str) -> str:
    """Return the packaged filename for an example name."""

    try:
        return EXAMPLE_FILES[name]
    except KeyError as exc:
        choices = ", ".join(available_examples())
        raise ValueError(f"unknown example {name!r}; choose one of: {choices}") from exc


def example_text(name: str) -> str:
    """Return the text for a packaged example."""

    filename = example_filename(name)
    return (
        resources.files("tokamaker_jax.example_data").joinpath(filename).read_text(encoding="utf-8")
    )


def write_example(name: str, output: str | Path | None = None, *, force: bool = False) -> Path:
    """Write a packaged example to ``output`` and return the path."""

    filename = example_filename(name)
    path = Path(filename if output is None else output)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(example_text(name), encoding="utf-8")
    return path.resolve()
