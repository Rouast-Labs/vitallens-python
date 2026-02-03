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
