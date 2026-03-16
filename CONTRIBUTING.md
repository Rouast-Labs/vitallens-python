# Contributing to vitallens-python

Thank you for your interest in contributing! This guide will help you set up your development environment and run the test suite.

## Development setup

We recommend using a virtual environment to manage dependencies.

### Clone and install

Clone the repository and install the package in **editable mode** with development dependencies.

```bash
# Clone the repo
git clone https://github.com/Rouast-Labs/vitallens-python.git
cd vitallens-python

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode
# This allows you to modify source code and see changes immediately.
pip install -e .

# Install dev tools
pip install flake8 pytest build
```

## Running tests

The test suite requires a valid API Key to verify integration with the live API.

1. **Set the Environment Variable:**

You must set `VITALLENS_DEV_API_KEY` before running tests.

```bash
# Mac/Linux
export VITALLENS_DEV_API_KEY="your_api_key_here"

# Windows (PowerShell)
$env:VITALLENS_DEV_API_KEY="your_api_key_here"
```

2. **Run Pytest:**

```bash
pytest
```

## Linting

We use `flake8` to ensure code quality. Please check your code before submitting a PR.

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Building the package

To build the distribution archives (source and wheel) for PyPI:

```bash
python -m build
```

The artifacts will be generated in the `dist/` directory.

Since this is a Python project using `setuptools_scm`, versioning is handled automatically based on git tags. I've adapted the workflow from your JS library to Python conventions while maintaining the same straightforward structure.

Add this section to the end of your `CONTRIBUTING.md`:

---

## Releases

Releases are automated via GitHub Actions and triggered by pushing git tags. The project uses `setuptools_scm`, so the version is derived directly from the tag name.

### Prerelease (Beta)

Use this to test new features without affecting stable users on PyPI.

1. Ensure your changes are merged into `main`.
2. Switch to `main` locally and pull the latest:
  ```bash
  git switch main
  git pull origin main
  ```
3. Create and push a beta tag (e.g., `v1.2.0-beta.1`):
  ```bash
  git tag -a v1.2.0-beta.1 -m "v1.2.0-beta.1"
  git push origin v1.2.0-beta.1
  ```

The CI will automatically publish to PyPI and create a GitHub prerelease.

### Production Release

1. Ensure the `main` branch is ready for production.
2. Switch to `main` locally and pull the latest:
  ```bash
  git switch main
  git pull origin main
  ```
3. Create and push a production tag (must start with `v`):
  ```bash
  git tag -a v1.2.0 -m "v1.2.0"
  git push origin v1.2.0
  ```

The CI will build the distribution archives, publish to PyPI, and create a stable GitHub Release with generated release notes.

### Syncing back to Dev

After any release, ensure the tags are available in your development branch to keep `setuptools_scm` versions in sync:

```bash
git switch dev
git merge main
git push origin dev
```
