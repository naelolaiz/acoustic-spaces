"""The graphical editor: build a room, place sources and listeners, assign materials,
run the simulation and look at and listen to the results.

Start it with ``acoustic-spaces edit [scene.yaml]`` or ``acoustic-spaces-editor``. It
needs the ``gui`` extra: ``pip install "acoustic-spaces[gui]"``, or
``uv sync --extra gui`` in a source checkout.
"""

from __future__ import annotations

from acoustic_spaces._extras import install_hint

INSTALL_HINT = f"the editor needs the gui extra: {install_hint('gui')}"


def main(argv: list[str] | None = None) -> int:
    """Run the editor, opening the scene file in ``argv`` if one is given."""
    try:
        import PySide6  # noqa: F401
        import pyvistaqt  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"{INSTALL_HINT} ({exc})") from exc
    from acoustic_spaces.gui.app import run

    return run(argv)


__all__ = ["INSTALL_HINT", "main"]
