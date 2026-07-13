import pandas as pd


def _pilot_profiles():
    cells = pd.read_csv("results/corrected_v2/pilot_protocol_v2_cells.csv")
    return cells.groupby("mechanism", as_index=False).agg(
        detectability=("diagnostic_normalized_ap", "mean"),
        exploitability=("paired_harm", "mean"),
    ).set_index("mechanism")


def test_all_eleven_constructed_contaminations_have_profiles():
    profiles = _pilot_profiles()
    assert set(profiles.index) == {f"M{i:02d}" for i in range(1, 12)}


def test_pilot_contains_crossed_detectability_exploitability_examples():
    profiles = _pilot_profiles()
    assert profiles.loc["M03", "detectability"] < 0.30
    assert profiles.loc["M03", "exploitability"] > 0.10
    assert profiles.loc["M08", "detectability"] > 0.30
    assert profiles.loc["M08", "exploitability"] < 0.02


def test_three_axes_are_not_encoded_as_one_scalar():
    profiles = _pilot_profiles()
    correlation = profiles["detectability"].corr(profiles["exploitability"], method="spearman")
    assert abs(correlation) < 0.9
