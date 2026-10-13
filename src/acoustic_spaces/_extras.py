"""Install hints for the optional extras, for error messages."""


def install_hint(extra: str) -> str:
    """How to install an optional extra, with pip or, in a source checkout, with uv."""
    return (
        f'pip install "acoustic-spaces[{extra}]", or uv sync --extra {extra} in a source checkout'
    )
