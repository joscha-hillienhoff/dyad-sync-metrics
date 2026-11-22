import pytest

from dyad_sync_metrics.io_utils import ensure_no_clobber


def test_passes_when_nothing_exists(tmp_path):
    ensure_no_clobber(tmp_path, ["a.csv", "b.csv"])
    ensure_no_clobber(tmp_path / "missing", ["a.csv"])


def test_refuses_and_leaves_existing_files_untouched(tmp_path):
    target = tmp_path / "a.csv"
    target.write_text("existing")
    with pytest.raises(FileExistsError, match="a.csv"):
        ensure_no_clobber(tmp_path, ["a.csv", "b.csv"])
    assert target.read_text() == "existing"


def test_overwrite_is_an_explicit_choice(tmp_path):
    (tmp_path / "a.csv").write_text("existing")
    ensure_no_clobber(tmp_path, ["a.csv"], overwrite=True)
