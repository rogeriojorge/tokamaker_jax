# Release and PyPI Publishing

This project publishes `tokamaker-jax` to PyPI from GitHub Releases. The
publish workflow builds the source distribution and wheel, then uses PyPI
Trusted Publishing; no long-lived PyPI token should be stored in the
repository.

## Before tagging

1. Confirm the version in `pyproject.toml` and `docs/conf.py` is the intended
   release version.
2. Install the release toolchain in a fresh environment:

   ```bash
   python -m pip install -U pip
   python -m pip install -e ".[dev,docs]"
   python -m pip install build twine
   ```

3. Run the local release checks:

   ```bash
   python -m ruff format --check .
   python -m ruff check .
   python -m pytest --cov=tokamaker_jax --cov-fail-under=95
   python -m sphinx -W -b html docs docs/_build/html
   rm -rf dist
   python -m build --sdist --wheel
   python -m twine check --strict dist/*
   ```

4. For physics-release candidates, also run the implemented validation gates:

   ```bash
   tokamaker-jax verify --gate all --subdivisions 4 8 16
   ```

## Trusted Publishing setup

PyPI must have a Trusted Publisher entry for this repository before the first
release can publish:

- PyPI project: `tokamaker-jax`
- Owner/repository: `rogeriojorge/tokamaker_jax`
- Workflow file: `publish.yml`
- Environment: `pypi`

The GitHub workflow runs release tests/docs, builds and checks the source
distribution and wheel, installs the built wheel as a smoke test, uploads those
exact distributions as an artifact, then publishes from the protected `pypi`
environment. The publish job needs `id-token: write` and `contents: read`
permissions so `pypa/gh-action-pypi-publish` can request an OpenID Connect token
from GitHub.

## Tag and publish

1. Create and push a signed or annotated tag that matches the package version:

   ```bash
   git tag -a v0.1.0a0 -m "tokamaker-jax 0.1.0a0"
   git push origin v0.1.0a0
   ```

2. Draft a GitHub Release from the tag. Include the validation commands that
   passed, the Python versions covered by CI, and any known feature-parity
   limits.
3. Publish the GitHub Release. The PyPI workflow runs on the `release:
   published` event and uploads `dist/*` to PyPI.

If a build fails before upload, fix the issue and publish a new release from a
new version tag. Do not reuse a PyPI version after files have been uploaded.

## Post-release verification

After the workflow succeeds:

```bash
python -m pip install --upgrade --force-reinstall tokamaker-jax
tokamaker-jax --help
tokamaker-jax init-example fixed-boundary --output fixed_boundary.toml
tokamaker-jax fixed_boundary.toml --plot outputs/fixed_boundary.png
tokamaker-jax verify --gate circular-loop
python - <<'PY'
from importlib.metadata import version

print(version("tokamaker-jax"))
PY
```

Then confirm:

- <https://pypi.org/project/tokamaker-jax/> shows the new version and files.
- The GitHub Release links to the expected tag and workflow run.
- Read the Docs has built the release or latest documentation.
- The README PyPI badge resolves for the published project.
