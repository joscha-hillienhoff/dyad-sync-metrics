"""Small file-handling helpers."""

from collections.abc import Iterable
from pathlib import Path


def ensure_no_clobber(out_dir: Path, names: Iterable[str], overwrite: bool = False) -> None:
    """Raise FileExistsError if any of names already exists in out_dir, unless
    overwrite is set.

    Call it before computing anything, so that a run never replaces earlier results by
    accident and never fails halfway through writing; overwriting is an explicit choice.
    """
    if overwrite:
        return
    existing = sorted(n for n in names if (Path(out_dir) / n).exists())
    if existing:
        raise FileExistsError(
            f"output file(s) already exist in {out_dir}: {', '.join(existing)}; choose an "
            "empty directory or pass overwrite=True"
        )
