# Getting Started

Install from a local checkout:

```bash
git clone https://github.com/rogeriojorge/tokamaker_jax.git
cd tokamaker_jax
pip install -e .
```

The default install includes GUI dependencies.

For development and docs work:

```bash
pip install -e ".[dev,docs]"
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
