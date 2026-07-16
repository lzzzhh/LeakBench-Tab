from pathlib import PurePosixPath

from paper.edbt_eab.source_data import build_edbt_draft_package as builder


def test_edbt_draft_package_is_explicit_and_excludes_historical_material():
    assert len(builder.ALLOWLIST) == len(set(builder.ALLOWLIST))
    assert PurePosixPath("paper/edbt_eab/main.tex") in builder.ALLOWLIST
    assert PurePosixPath("results/corrected_v2/canonical_cells.csv") in builder.ALLOWLIST
    assert PurePosixPath("artifacts/sp8/governance_clean.csv") in builder.ALLOWLIST
    assert all("pilot" not in path.parts for path in builder.ALLOWLIST)


def test_edbt_draft_package_remains_explicitly_non_final():
    assert builder.DESTINATION.name.endswith("_draft")
    assert builder.ARCHIVE.name.endswith("_draft.zip")
