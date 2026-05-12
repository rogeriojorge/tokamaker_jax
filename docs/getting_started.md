# Getting Started

Install from a local checkout:

```bash
pip install -e ".[dev,gui,docs]"
```

Launch the GUI:

```bash
tokamaker-jax
```

Run a TOML configuration:

```bash
tokamaker-jax examples/fixed_boundary.toml --plot outputs/fixed_boundary.png
```

Python 3.10 is supported. TOML parsing uses `tomli` on Python 3.10 and `tomllib` on Python 3.11+.

