from pathlib import PurePosixPath

from paper.edbt_eab.source_data import build_edbt_source_package as builder


def test_source_package_contains_every_direct_compile_input():
    expected = {
        PurePosixPath("main.tex"),
        PurePosixPath("references.bib"),
        PurePosixPath("acmart.cls"),
        PurePosixPath("ACM-Reference-Format.bst"),
        PurePosixPath("edbt-macros.tex"),
        PurePosixPath("generated/result_macros.tex"),
        PurePosixPath("generated/table_measurement.tex"),
        PurePosixPath("generated/table_natural.tex"),
        PurePosixPath("generated/table_governance.tex"),
        PurePosixPath("figures/generated/cdx_profiles.pdf"),
        PurePosixPath("figures/generated/governance_tradeoff.pdf"),
    }
    assert expected.issubset(set(builder.SOURCE_FILES))


def test_source_package_paths_are_repository_independent():
    assert all(not path.is_absolute() for path in builder.SOURCE_FILES)
    assert all(".." not in path.parts for path in builder.SOURCE_FILES)
    assert all("paper" not in path.parts for path in builder.SOURCE_FILES)
