import pytest, numpy as np

PROFILES = {
    "M01": "C1-DH-XH", "M02": "C1-DH-XH", "M03": "C1-DH-XL",
    "M04": "C1-DL-XL", "M05": "C1-DL-XL", "M06": "C1-DH-XH",
    "M07": "C1-DM-XC", "M08": "C1-DL-XC", "M09": "C1-DL-XL",
    "M10": "C1-DH-XH", "M11": "C1-DH-XH",
}

def test_all_11_mechanisms_profiled():
    assert len(PROFILES) == 11

def test_profile_format():
    for mid, profile in PROFILES.items():
        parts = profile.split("-")
        assert len(parts) == 3
        assert parts[0] in ("C0", "C1")
        assert parts[1] in ("DH", "DM", "DL")
        assert parts[2] in ("XH", "XC", "XL")

def test_simple_all_dh():
    for mid in ["M01","M02","M06","M10"]:
        assert PROFILES[mid].startswith("C1-DH")

def test_structured_all_dl():
    for mid in ["M04","M05","M08","M09"]:
        assert PROFILES[mid].startswith("C1-DL")

def test_no_contamination_zero():
    for mid in PROFILES:
        assert PROFILES[mid].startswith("C1")  # All mechanisms are contaminated by construction