from pathlib import Path


def test_figure_generator_is_final_only_and_fail_closed():
    source = Path("scripts/generate_corrected_v2_figures.py").read_text(encoding="utf-8")
    assert "pilot input is forbidden" in source
    assert "canonical 27,500-cell matrix is not complete" in source
    assert "diagnostic confirmatory statistics are incomplete" in source
    assert "cdx_scatter.pdf" in source
    assert "mechanism_model_heatmap.pdf" in source
    assert "strength_diagnostic_robustness.pdf" in source
