"""Install hints for the optional extras, for error messages."""

from __future__ import annotations

from importlib import metadata


def _requirements(extra: str) -> list[str]:
    """The packages an extra adds, as the installed package's metadata lists them."""
    try:
        requires = metadata.requires("acoustic-spaces") or []
    except metadata.PackageNotFoundError:
        return []
    out = []
    for requirement in requires:
        name, _, marker = requirement.partition(";")
        if f'extra == "{extra}"' in marker.replace("'", '"'):
            out.append(name.strip())
    return out


def install_hint(extra: str) -> str:
    """How to install an optional extra: the packages it adds with pip (the package is
    not on PyPI, so ``pip install "acoustic-spaces[extra]"`` would look for it there), or,
    in a source checkout, with uv."""
    packages = _requirements(extra)
    pip = (
        "pip install " + " ".join(f'"{p}"' for p in packages)
        if packages
        else f"install the {extra} extra"
    )
    return f"{pip}, or uv sync --extra {extra} in a source checkout"
