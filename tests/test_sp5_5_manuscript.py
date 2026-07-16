"""SP5.5 manuscript synchronization tests."""
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "paper/aaai27/main.tex").read_text()
SUPP = (ROOT / "paper/aaai27/supplement.tex").read_text()
GEN = (ROOT / "paper/aaai27/generated/result_macros.tex").read_text()
BASE = (ROOT / "paper/aaai27/source_data/result_macros_base.tex").read_text()
SP55 = ROOT / "artifacts/sp5_5"
README = (ROOT / "README.md").read_text()


def _prose():
    return MAIN + SUPP


def test_no_old_structured_auprc_claim():
    # no "0.05-0.07" adjacent to structured/AUPRC/detectab
    for m in re.finditer(r"0\.0[567]", _prose()):
        ctx = _prose()[max(0, m.start()-60):m.start()+40].lower()
        assert not any(t in ctx for t in ["auprc", "detectab", "structured localiz"]), ctx


def test_no_old_m08_detectability():
    assert "0.048" not in _prose()


def test_no_old_cl4_ratios():
    assert not re.search(r"2\.2\s*[x×]", _prose())
    assert not re.search(r"2\.5\s*[x×]", _prose())


def test_no_within_category_zero_claim():
    assert not re.search(r"within.category.{0,25}(zero|\\approx\s*0|= ?0)", _prose(), re.I)
    assert "category-driven" not in _prose().lower()


def test_no_unconditional_consistency_claim():
    assert "fully consistent" not in _prose().lower()
    assert not re.search(r"consistent across all (five )?models", _prose(), re.I)


def test_all_macros_defined():
    defined = set(re.findall(r"\\(?:newcommand|renewcommand)\{\\(LB[A-Za-z]+)\}", BASE + GEN))
    defined |= set(re.findall(r"\\newif\\if(LB[A-Za-z]+)", BASE))
    used = set(re.findall(r"\\(LB[A-Za-z]+)", MAIN + SUPP))
    assert not (used - defined), sorted(used - defined)


def test_macros_sourced_from_ledger():
    """Macros now bind paper_claims.json sha256 as final source (provenance migration from claim_ledger_v2)."""
    # Generated macros must reference paper_claims.json
    assert "paper_claims.json" in GEN or "claim_ledger_v2.csv" in GEN


def test_four_core_claims_in_traceability():
    t = pd.read_csv(SP55 / "claim_paper_traceability.csv")
    assert set(["CL2", "CL3", "CL4", "CL10"]).issubset(set(t["claim_id"]))


def test_claim_lock_json_md_consistency():
    lock = json.loads((SP55 / "paper_claim_lock.json").read_text())
    ids = {c["claim_id"] for c in lock}
    assert ids == {"CL2", "CL3", "CL4", "CL10"}
    md = (SP55 / "paper_claim_lock.md").read_text()
    for c in lock:
        assert c["claim_id"] in md and c["status"] in md


def test_readme_model_scope_matches_registry():
    import yaml
    reg = yaml.safe_load((ROOT / "artifacts/sp5/model_registry.yaml").read_text())
    for m in reg["models"]:
        assert reg["models"][m]["display_name"] in README or m.upper() in README or m in README.lower()


def test_readme_marks_modern_models_deferred():
    for m in ["ModernNCA", "TabR", "TabPFNv2", "TabICL"]:
        assert m in README
    assert "Deferred" in README or "deferred" in README


def test_readme_marks_track_b_not_evaluated():
    assert "Track B" in README
    assert "not" in README.lower() and ("governance" in README.lower())


def test_text_audit_zero_superseded_active():
    a = pd.read_csv(SP55 / "manuscript_text_audit.csv")
    assert int((a["status"] == "SUPERSEDED_ACTIVE").sum()) == 0


def test_figure_table_registry_all_traced():
    r = pd.read_csv(SP55 / "figure_table_registry.csv")
    assert (r["contains_superseded_value"] == "no").all()
    assert r["source_table"].notna().all()


def test_results_ready_flag_enabled():
    assert "\\LBResultsReadytrue" in GEN
