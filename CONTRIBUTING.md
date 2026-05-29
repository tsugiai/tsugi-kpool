# Contributing to tsugi-kpool

Thank you for contributing to `tsugi-kpool`. This project is pre-alpha, so small
focused pull requests with clear tests are easiest to review.

## Development Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Optional benchmark dependencies are not required for ordinary development. Only
install them when working on benchmark-specific changes:

```bash
pip install -e ".[benchmark]"
```

## Checks

Run the standard checks before opening a pull request:

```bash
ruff check src tests
mypy src
pytest -q
```

When a change affects examples or documented usage, also run the relevant
example or document why it cannot be run locally. Some examples require a gated
model, Hugging Face authentication, or GPU hardware.

## Branch and Pull Request Conventions

- Create a topic branch from the latest `main`.
- Keep each pull request focused on one logical change.
- Add or update tests for behavior changes.
- Do not commit secrets, tokens, private paths, or environment-specific files.
- Keep public APIs stable unless the pull request clearly explains a breaking
  change.

Pull requests should include:

- What changed.
- Why the change is needed.
- Test evidence for `ruff`, `mypy`, `pytest`, and any examples run.
- Risk and review notes, including dependency or workflow changes.

Maintainers review pull requests before merging and generally squash-merge
accepted changes.

## Reporting Issues

Use GitHub issues for bugs and feature requests. For security vulnerabilities,
follow `SECURITY.md` and report privately instead of opening a public issue.
