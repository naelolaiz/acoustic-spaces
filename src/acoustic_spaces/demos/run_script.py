"""Runs an example's script that is no demo (:func:`~acoustic_spaces.demos.script_demo`)
as the editor's Run demo does::

    python -m acoustic_spaces.demos.run_script SCRIPT [ARGUMENTS...]

runs ``SCRIPT`` as ``python SCRIPT [ARGUMENTS...]`` would (its folder first on the
module path, its name as the command's), then, if it ended well, prints the line that
begins with :data:`~acoustic_spaces.demos.DONE`. From that line the editor knows that
what the script wrote is complete, whatever its process does next, as it knows it of a
demo (:meth:`~acoustic_spaces.demos.Demo.run`)."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

from acoustic_spaces.demos import DONE


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: python -m acoustic_spaces.demos.run_script SCRIPT [ARGUMENTS...]",
            file=sys.stderr,
        )
        return 2
    script = Path(argv[0]).resolve()
    sys.argv = [str(script), *argv[1:]]
    sys.path[0] = str(script.parent)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as stop:
        if stop.code not in (None, 0):
            raise
    print(f"{DONE}: {script.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
