from pathlib import Path

import pytest

from scripts import build_aaai27_paper as paper_build


def test_parse_main_content_page_uses_page_component():
    aux = r"\newlabel{lb:last-main-content-page}{{7}{4}{}{section.7}{}}"
    assert paper_build.parse_main_content_page(aux) == 4


def test_parse_fonts_recovers_embedding_and_type_columns():
    output = """\
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
ABCDEF+Termes                        Type 1            Custom           yes yes yes      4  0
GHIJKL+Modern                        Type 3            Builtin          no  no  yes      5  0
"""
    fonts = paper_build.parse_fonts(output)
    assert fonts[0]["type"] == "Type 1"
    assert fonts[0]["embedded"] == "yes"
    assert fonts[1]["type"] == "Type 3"
    assert fonts[1]["embedded"] == "no"


def test_final_log_rejects_overfull_box_but_draft_can_report_it(tmp_path: Path):
    log = tmp_path / "main.log"
    log.write_text("Overfull \\vbox (1.0pt too high) has occurred\n", encoding="utf-8")
    with pytest.raises(paper_build.BuildError, match="overfull_box"):
        paper_build.validate_log(log)
    result = paper_build.validate_log(log, allow_overfull=True)
    assert result["overfull_boxes"] is False


def test_final_input_gate_fails_before_pdf_build_when_claims_are_absent():
    if (paper_build.ROOT / "results/corrected_v2/paper_claims.json").is_file():
        pytest.skip("Final claim release is present in this checkout")
    with pytest.raises(paper_build.BuildError, match="claims/macros are absent"):
        paper_build.verify_final_inputs()
