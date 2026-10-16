"""Install hints for the optional extras, for error messages."""

from __future__ import annotations


def install_hint(extra: str) -> str:
    """How to install an optional extra: with pip, which adds only the extra's packages to
    an acoustic-spaces already installed (from PyPI, a release's wheel or a source
    checkout), or, in a source checkout, with uv."""
    return (
        f'pip install "acoustic-spaces[{extra}]", or uv sync --extra {extra} in a source checkout'
    )
