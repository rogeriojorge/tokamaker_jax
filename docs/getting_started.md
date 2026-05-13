# Getting Started

Install from a local checkout:

```bash
git clone https://github.com/rogeriojorge/tokamaker_jax.git
cd tokamaker_jax
pip install -e .
```

The default install includes GUI dependencies.

After PyPI publication, a wheel-only install can create a runnable local example
without cloning the repository:

```bash
pip install tokamaker-jax
tokamaker-jax init-example fixed-boundary --output fixed_boundary.toml
tokamaker-jax fixed_boundary.toml --plot outputs/fixed_boundary.png
```

For development and docs work:

```bash
pip install -e ".[dev,docs]"
```

Launch the GUI:

```bash
tokamaker-jax
```

Or choose an explicit host/port, which is useful on remote Linux sessions, WSL,
containers, or when port 8080 is already in use:

```bash
tokamaker-jax gui --host 127.0.0.1 --port 8081 --no-browser
```

Run a TOML configuration:

```bash
tokamaker-jax examples/fixed_boundary.toml --plot outputs/fixed_boundary.png
```

Python 3.10 is supported. TOML parsing uses `tomli` on Python 3.10 and `tomllib` on Python 3.11+.
