#!/usr/bin/env python3
"""Generate complete, claim-scoped supplement tables from final evidence.

No pilot path is accepted. Every consumed statistics file must be present in
the final claim provenance or in a separately hash-frozen task/public-natural
manifest. The output is deterministic LaTeX plus a source/output hash
manifest consumed by the final PDF build gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "paper/aaai27/generated"
CLAIMS = ROOT / "results/corrected_v2/paper_claims.json"
STATS = ROOT / "results/corrected_v2/statistics"
TASK_MANIFEST = ROOT / "results/corrected_v2/task_bundles/task_manifest.csv"
STAT_FREEZE = ROOT / "results/corrected_v2/statistical_amendment_protocol_v2_freeze.json"
PUBLIC_NATURAL = ROOT / "results/corrected_v2/public_natural"

MECHANISMS = [f"M{index:02d}" for index in range(1, 12)]
MODELS = ["lr", "rf", "catboost", "lightgbm", "tabm"]
METHODS = [
    "mutual_information",
    "absolute_correlation",
    "lr_coefficient",
    "rf_permutation",
]
CATEGORIES = {
    "M01": "simple", "M02": "simple", "M03": "boundary",
    "M04": "structured", "M05": "structured", "M06": "simple",
    "M07": "boundary", "M08": "structured", "M09": "structured",
    "M10": "simple", "M11": "boundary",
}
NATURAL_BOUNDARIES = {
    "BankMarketing": "Before call completion",
    "LendingClub": "At loan origination",
    "BTSFlights": "Before scheduled departure",
    "ChicagoFood": "Before inspection outcome",
    "NYC311": "At complaint creation",
}


class TableError(ValueError):
    """Raised when complete final tables cannot be safely rendered."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise TableError(f"Required file is missing: {relative(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TableError(f"JSON root is not an object: {relative(path)}")
    return value


def provenance(claims: Mapping[str, Any]) -> Mapping[str, str]:
    block = claims.get("provenance")
    if not isinstance(block, Mapping) or not isinstance(block.get("input_sha256"), Mapping):
        raise TableError("paper_claims.json has no final provenance hash map")
    return block["input_sha256"]


def require_claim_source(path: Path, hashes: Mapping[str, str]) -> None:
    name = relative(path)
    if hashes.get(name) != sha256(path):
        raise TableError(f"Table source is absent or stale in claim provenance: {name}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise TableError(f"{label} is missing columns: {sorted(missing)}")


def tex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def number(value: Any, digits: int = 3) -> str:
    observed = float(value)
    if not math.isfinite(observed):
        raise TableError(f"Cannot format non-finite value: {value!r}")
    if abs(observed) < 0.5 * 10 ** (-digits):
        observed = 0.0
    return f"{observed:.{digits}f}"


def pvalue(value: Any) -> str:
    observed = float(value)
    if not 0.0 <= observed <= 1.0:
        raise TableError(f"Invalid p-value: {value!r}")
    if 0.0 < observed < 0.0001:
        return f"{observed:.2e}"
    return f"{observed:.4f}"


def interval(point: Any, low: Any, high: Any) -> str:
    point_value, low_value, high_value = map(float, (point, low, high))
    if not low_value <= point_value <= high_value:
        raise TableError(
            f"Unordered interval: {low_value} <= {point_value} <= {high_value}"
        )
    return f"{number(point_value)} [{number(low_value)}, {number(high_value)}]"


def table_environment(
    *,
    columns: str,
    header: str,
    rows: Iterable[str],
    caption: str,
    label: str,
    tabcolsep: str = "3.2pt",
) -> str:
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[p]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{{tabcolsep}}}
\\begin{{tabular}}{{{columns}}}
\\toprule
{header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table*}}
\\clearpage
"""


def validate_claims() -> Mapping[str, Any]:
    claims = load_json(CLAIMS)
    protocol = claims.get("protocol_integrity", {})
    if (
        claims.get("evidence_tier") != "confirmatory"
        or claims.get("release_status") != "CLAIM_STATE_DERIVED"
        or protocol.get("expected_cells") != 27_500
        or protocol.get("successful_cells") != 27_500
        or protocol.get("completion_rate") != 1.0
        or set(protocol.get("models", [])) != set(MODELS)
    ):
        raise TableError("paper_claims.json is not the complete final confirmatory release")
    claim_block = claims.get("claims", {})
    expected_claims = {
        "simple_vs_structured", "m03_profile", "m08_profile",
        "m09_counterexample", "detectability_exploitability_relation",
        "D_METHOD_CONDITIONAL",
    }
    if set(claim_block) != expected_claims:
        raise TableError("Final paper claim identity set changed")
    if claim_block["simple_vs_structured"].get("status") not in {
        "SUPPORTED", "NOT_SUPPORTED"
    }:
        raise TableError("Directional category claim has no final decision")
    for claim_id in expected_claims - {"simple_vs_structured"}:
        if claim_block[claim_id].get("status") != "DESCRIPTIVE_ONLY":
            raise TableError(f"{claim_id} exceeded its descriptive scope cap")
    if claims.get("natural", {}).get("status") != "CASE_STUDY_ONLY":
        raise TableError("Natural evidence exceeded its case-study scope cap")
    return claims


def task_registry() -> tuple[str, list[Path]]:
    freeze = load_json(STAT_FREEZE)
    frozen = freeze.get("frozen_files", {})
    manifest_name = relative(TASK_MANIFEST)
    if (
        not isinstance(frozen, Mapping)
        or frozen.get(manifest_name, {}).get("sha256") != sha256(TASK_MANIFEST)
    ):
        raise TableError("Task manifest is not bound by the v2 statistical freeze")
    bundle_hashes = freeze.get("bundle_sha256_by_path")
    if not isinstance(bundle_hashes, Mapping) or len(bundle_hashes) != 20:
        raise TableError("The v2 freeze does not bind exactly twenty task bundles")

    frame = pd.read_csv(TASK_MANIFEST)
    required = [
        "dataset_id", "dataset_index", "dataset_seed", "archetype", "n_samples",
        "n_original", "split_hash", "bundle_path", "bundle_sha256",
    ]
    require_columns(frame, required, "task_manifest")
    if len(frame) != 5_500 or frame["dataset_id"].nunique() != 20:
        raise TableError("Task manifest does not contain the frozen 5,500 tasks")

    rows: list[str] = []
    sources = [TASK_MANIFEST, STAT_FREEZE]
    for dataset_id, group in frame.groupby("dataset_id", sort=True):
        for column in required[1:]:
            if group[column].nunique(dropna=False) != 1:
                raise TableError(f"{dataset_id}.{column} is not invariant")
        entry = group.iloc[0]
        bundle = ROOT / str(entry["bundle_path"])
        if not bundle.is_file() or sha256(bundle) != str(entry["bundle_sha256"]):
            raise TableError(f"Task bundle hash mismatch for {dataset_id}")
        if bundle_hashes.get(relative(bundle)) != sha256(bundle):
            raise TableError(f"Task bundle is not v2-freeze-bound for {dataset_id}")
        sources.append(bundle)
        with np.load(bundle, allow_pickle=False) as data:
            needed = {
                "base_X", "y", "train_idx", "val_idx", "test_idx",
                "base_entity_ids", "source_ids__M09_S1_13",
            }
            if not needed <= set(data.files):
                raise TableError(f"Bundle {dataset_id} lacks registry arrays")
            n = len(data["y"])
            if n != int(entry["n_samples"]):
                raise TableError(f"Bundle/manifest row mismatch for {dataset_id}")
            split_sizes = tuple(
                len(data[name]) for name in ("train_idx", "val_idx", "test_idx")
            )
            expected_split = (int(0.6 * n), int(0.2 * n), n - int(0.8 * n))
            if split_sizes != expected_split:
                raise TableError(
                    f"Unexpected 60/20/20 split for {dataset_id}: {split_sizes}"
                )
            prevalence = float(np.mean(data["y"]))
            feature_count = int(data["base_X"].shape[1])
            entity_count = int(np.unique(data["base_entity_ids"]).size)
            source_count = int(np.unique(data["source_ids__M09_S1_13"]).size)
        rows.append(
            f"{tex(dataset_id)} & {tex(entry['archetype'])} & {n:,} & {feature_count} & "
            f"{number(prevalence)} & {entity_count} & {source_count} & "
            f"{int(entry['dataset_seed'])} & {tex(str(entry['split_hash'])[:10])} \\\\"
        )
    rendered = table_environment(
        columns="llrrrrrrl",
        header=(
            r"Task & Archetype & $n$ & $p$ & $\Pr(y{=}1)$ & Entities & "
            r"Sources & Seed & Split SHA"
        ),
        rows=rows,
        caption=(
            "Complete frozen controlled-task registry. Split hashes are shown as "
            "ten-character prefixes; the artifact manifest retains full SHA-256 values."
        ),
        label="tab:complete-task-registry",
        tabcolsep="3.0pt",
    )
    return rendered, sources


def mechanism_profiles(hashes: Mapping[str, str]) -> tuple[str, list[Path]]:
    harm_path = STATS / "mechanism_summary.csv"
    detect_path = STATS / "detectability_mechanism_summary.csv"
    for path in (harm_path, detect_path):
        require_claim_source(path, hashes)
    harm = pd.read_csv(harm_path)
    detect = pd.read_csv(detect_path)
    require_columns(
        harm,
        [
            "mechanism", "category", "paired_harm", "paired_harm_ci_low",
            "paired_harm_ci_high", "sign_flip_p", "holm_p",
        ],
        "mechanism_summary",
    )
    require_columns(
        detect,
        [
            "mechanism", "category", "diagnostic_normalized_ap",
            "diagnostic_normalized_ap_ci_low", "diagnostic_normalized_ap_ci_high",
        ],
        "detectability_mechanism_summary",
    )
    if (
        set(harm["mechanism"]) != set(MECHANISMS)
        or set(detect["mechanism"]) != set(MECHANISMS)
    ):
        raise TableError("Mechanism summary identity set changed")
    merged = harm.merge(detect, on=["mechanism", "category"], validate="one_to_one")
    rows = []
    for mechanism in MECHANISMS:
        row = merged.loc[merged["mechanism"] == mechanism].iloc[0]
        if row["category"] != CATEGORIES[mechanism]:
            raise TableError(f"Category changed for {mechanism}")
        rows.append(
            f"{mechanism} & {tex(row['category'].title())} & "
            f"{interval(row['diagnostic_normalized_ap'], row['diagnostic_normalized_ap_ci_low'], row['diagnostic_normalized_ap_ci_high'])} & "
            f"{interval(row['paired_harm'], row['paired_harm_ci_low'], row['paired_harm_ci_high'])} & "
            f"{pvalue(row['sign_flip_p'])} & {pvalue(row['holm_p'])} \\\\"
        )
    rendered = table_environment(
        columns="llrrrr",
        header=(
            r"Mechanism & Category & MI $D$ [95\% CI] & $X$ [95\% CI] & "
            r"Raw $p$ & Holm $p$"
        ),
        rows=rows,
        caption=(
            "Complete mechanism-level C/D/X summaries. D is normalized AP under the "
            "frozen MI diagnostic; X is paired AUROC harm. Mechanism tests and intervals "
            "are reported for completeness, but no mechanism receives a binary claim decision."
        ),
        label="tab:complete-mechanism-profiles",
    )
    return rendered, [harm_path, detect_path]


def mechanism_models(hashes: Mapping[str, str]) -> tuple[str, list[Path]]:
    path = STATS / "mechanism_model_summary.csv"
    require_claim_source(path, hashes)
    frame = pd.read_csv(path)
    require_columns(
        frame,
        ["mechanism", "category", "model", "paired_harm", "ci_low", "ci_high"],
        "mechanism_model_summary",
    )
    if (
        len(frame) != 55
        or set(frame["mechanism"]) != set(MECHANISMS)
        or set(frame["model"]) != set(MODELS)
        or frame.duplicated(["mechanism", "model"]).any()
    ):
        raise TableError("Mechanism-model table is not the complete 11 x 5 matrix")
    indexed = frame.set_index(["mechanism", "model"])
    rows = []
    for mechanism in MECHANISMS:
        cells = []
        for model in MODELS:
            row = indexed.loc[(mechanism, model)]
            cells.append(
                interval(row["paired_harm"], row["ci_low"], row["ci_high"])
            )
        rows.append(
            f"{mechanism} & {tex(CATEGORIES[mechanism].title())} & "
            + " & ".join(cells)
            + r" \\"
        )
    rendered = table_environment(
        columns="llrrrrr",
        header="Mechanism & Category & LR & RF & CatBoost & LightGBM & TabM",
        rows=rows,
        caption=(
            "Paired AUROC harm with 95\\% task-hierarchical intervals for all 55 fixed "
            "mechanism--model cells. Model-family contrasts are descriptive, not causal."
        ),
        label="tab:complete-mechanism-model",
        tabcolsep="2.5pt",
    )
    return rendered, [path]


def diagnostic_methods(hashes: Mapping[str, str]) -> tuple[str, list[Path]]:
    path = STATS / "diagnostic_method_by_mechanism.csv"
    require_claim_source(path, hashes)
    frame = pd.read_csv(path)
    require_columns(
        frame,
        [
            "method", "mechanism", "category", "diagnostic_normalized_ap",
            "ci_low", "ci_high",
        ],
        "diagnostic_method_by_mechanism",
    )
    if (
        len(frame) != 44
        or set(frame["mechanism"]) != set(MECHANISMS)
        or set(frame["method"]) != set(METHODS)
        or frame.duplicated(["mechanism", "method"]).any()
    ):
        raise TableError("Diagnostic table is not the complete 11 x 4 matrix")
    indexed = frame.set_index(["mechanism", "method"])
    rows = []
    for mechanism in MECHANISMS:
        cells = []
        for method in METHODS:
            row = indexed.loc[(mechanism, method)]
            cells.append(
                interval(
                    row["diagnostic_normalized_ap"], row["ci_low"], row["ci_high"]
                )
            )
        rows.append(
            f"{mechanism} & {tex(CATEGORIES[mechanism].title())} & "
            + " & ".join(cells)
            + r" \\"
        )
    rendered = table_environment(
        columns="llrrrr",
        header="Mechanism & Category & MI & Abs. corr. & LR coef. & RF perm.",
        rows=rows,
        caption=(
            "Normalized-AP localization with 95\\% task-hierarchical intervals under all "
            "four frozen oracle-blind diagnostics. MI is primary; cross-method comparisons "
            "are descriptive sensitivity analyses and the row maximum is not a selector."
        ),
        label="tab:complete-diagnostic-methods",
        tabcolsep="3.0pt",
    )
    return rendered, [path]


def strength_response(hashes: Mapping[str, str]) -> tuple[str, list[Path]]:
    path = STATS / "strength_dose_response.csv"
    require_claim_source(path, hashes)
    frame = pd.read_csv(path)
    required = [
        "mechanism", "category", "standardized_strength_slope", "ci_low", "ci_high",
        "positive_adjacent_steps", "total_adjacent_steps",
        *[f"harm_S{index}" for index in range(1, 6)],
    ]
    require_columns(frame, required, "strength_dose_response")
    if len(frame) != 11 or set(frame["mechanism"]) != set(MECHANISMS):
        raise TableError("Strength-response table is incomplete")
    rows = []
    indexed = frame.set_index("mechanism")
    for mechanism in MECHANISMS:
        row = indexed.loc[mechanism]
        strengths = " & ".join(
            number(row[f"harm_S{index}"]) for index in range(1, 6)
        )
        slope = interval(
            row["standardized_strength_slope"], row["ci_low"], row["ci_high"]
        )
        steps = f"{int(row['positive_adjacent_steps'])}/{int(row['total_adjacent_steps'])}"
        rows.append(
            f"{mechanism} & {tex(CATEGORIES[mechanism].title())} & "
            f"{strengths} & {slope} & {steps} \\\\"
        )
    rendered = table_environment(
        columns="llrrrrrrr",
        header=(
            r"Mechanism & Category & S1 & S2 & S3 & S4 & S5 & "
            r"Std. slope [95\% CI] & $+$/4"
        ),
        rows=rows,
        caption=(
            "Pre-ordered strength response averaged across models. Slopes and intervals are "
            "descriptive; post hoc bootstrap-tail areas are deliberately omitted because they "
            "are not frozen inferential p-values. The last column counts positive adjacent steps."
        ),
        label="tab:complete-strength-response",
        tabcolsep="2.5pt",
    )
    return rendered, [path]


def natural_cases() -> tuple[str, list[Path]]:
    manifest_path = PUBLIC_NATURAL / "public_natural_provenance_manifest.json"
    tasks_path = PUBLIC_NATURAL / "natural_task_summary.csv"
    stats_path = PUBLIC_NATURAL / "natural_statistics.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "PUBLIC_NATURAL_PROVENANCE_PROJECTED"
        or manifest.get("all_scientific_invariants_passed") is not True
        or manifest.get("raw_natural_data_included") is not False
    ):
        raise TableError("Public natural provenance projection is not valid")
    public_outputs = manifest.get("public_outputs", {})
    for logical, path in (("tasks", tasks_path), ("statistics", stats_path)):
        entry = public_outputs.get(logical, {})
        if entry.get("path") != relative(path) or entry.get("sha256") != sha256(path):
            raise TableError(f"Public natural {logical} hash mismatch")
    tasks = pd.read_csv(tasks_path)
    stats = load_json(stats_path)
    expected = set(NATURAL_BOUNDARIES)
    if (
        len(tasks) != 5
        or set(tasks["task"]) != expected
        or set(stats.get("task_effects", {})) != expected
    ):
        raise TableError("Natural-case identity set changed")
    require_columns(
        tasks,
        [
            "task", "n_samples", "n_features", "n_leak",
            "diagnostic_normalized_ap", "source", "source_sha256",
            "preprocessing_protocol",
        ],
        "public natural task summary",
    )
    rows = []
    indexed = tasks.set_index("task")
    for task in (
        "BankMarketing", "LendingClub", "BTSFlights", "ChicagoFood", "NYC311"
    ):
        row = indexed.loc[task]
        if row["preprocessing_protocol"] != "natural_trainfit_categories_v2":
            raise TableError(f"Natural preprocessing protocol changed for {task}")
        if str(row["source"]).startswith(("/", "~")) or ":\\" in str(row["source"]):
            raise TableError(f"Public natural source path is not redacted for {task}")
        rows.append(
            f"{tex(task)} & {tex(NATURAL_BOUNDARIES[task])} & "
            f"{int(row['n_samples']):,} & {int(row['n_features'])} & "
            f"{int(row['n_leak'])} & {number(row['diagnostic_normalized_ap'])} & "
            f"{number(stats['task_effects'][task])} & "
            f"{tex(str(row['source_sha256'])[:10])} \\\\"
        )
    rendered = table_environment(
        columns="llrrrrrl",
        header=(
            r"Case & Prediction boundary & $n$ & $p$ & Leak & MI $D$ & "
            r"Mean $X$ & Source SHA"
        ),
        rows=rows,
        caption=(
            "Five fixed real-data boundary audits under train-fitted categorical "
            "preprocessing. Source hashes are ten-character prefixes of the full public "
            "lineage hashes. Mean X is descriptive across four CPU learners and three "
            "seeds; these are not a task-population sample."
        ),
        label="tab:complete-natural-cases",
        tabcolsep="2.8pt",
    )
    return rendered, [manifest_path, tasks_path, stats_path]


def claim_scope(claims: Mapping[str, Any]) -> str:
    block = claims["claims"]
    entries = [
        (
            "simple_vs_structured", block["simple_vs_structured"]["status"],
            "Category contrast", "Binary decision by exact Holm p and CI",
        ),
        (
            "m03_profile", block["m03_profile"]["status"], "M03 C/D/X profile",
            "Registry-bounded description",
        ),
        (
            "m08_profile", block["m08_profile"]["status"],
            "M08 synchronized entity interval", "No equivalence or practical-null claim",
        ),
        (
            "m09_counterexample", block["m09_counterexample"]["status"],
            "M09 encoded-column profile", "Representation-conditional description",
        ),
        (
            "D_X_relation", block["detectability_exploitability_relation"]["status"],
            "Eleven mechanism means", "No out-of-registry prediction",
        ),
        (
            "D_METHOD_CONDITIONAL", block["D_METHOD_CONDITIONAL"]["status"],
            "Four fixed diagnostics", "No inferential ranking or selector",
        ),
        (
            "natural_cases", claims["natural"]["status"], "Five fixed case studies",
            "No dataset-population inference",
        ),
    ]
    rows = [
        f"{tex(claim_id)} & {tex(status)} & {tex(evidence)} & {tex(scope)} \\\\"
        for claim_id, status, evidence, scope in entries
    ]
    return table_environment(
        columns="llll",
        header="Claim ID & Status & Evidence unit & Maximum interpretation",
        rows=rows,
        caption=(
            "Complete claim--evidence scope matrix. Only the directional category contrast "
            "has a thresholded support decision; all other controlled summaries are "
            "descriptive and natural evidence remains case-study-only."
        ),
        label="tab:complete-claim-scope",
        tabcolsep="4.0pt",
    )


def write_tables(claims: Mapping[str, Any]) -> dict[str, Any]:
    hashes = provenance(claims)
    outputs: dict[str, str] = {}
    sources: set[Path] = {CLAIMS, Path(__file__).resolve()}
    builders = [
        ("table_task_registry.tex", task_registry),
        ("table_mechanism_profiles.tex", lambda: mechanism_profiles(hashes)),
        ("table_mechanism_models.tex", lambda: mechanism_models(hashes)),
        ("table_diagnostic_methods.tex", lambda: diagnostic_methods(hashes)),
        ("table_strength_response.tex", lambda: strength_response(hashes)),
        ("table_natural_cases.tex", natural_cases),
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, builder in builders:
        rendered, used = builder()
        path = OUTPUT / filename
        path.write_text(
            "% Machine generated; do not edit.\n" + rendered, encoding="utf-8"
        )
        outputs[relative(path)] = sha256(path)
        sources.update(used)

    scope_path = OUTPUT / "table_claim_scope.tex"
    scope_path.write_text(
        "% Machine generated; do not edit.\n" + claim_scope(claims),
        encoding="utf-8",
    )
    outputs[relative(scope_path)] = sha256(scope_path)

    wrapper_path = OUTPUT / "result_tables.tex"
    wrapper_lines = [
        "% Machine generated; do not edit.",
        r"\input{generated/table_task_registry.tex}",
        r"\input{generated/table_mechanism_profiles.tex}",
        r"\input{generated/table_mechanism_models.tex}",
        r"\input{generated/table_diagnostic_methods.tex}",
        r"\input{generated/table_strength_response.tex}",
        r"\input{generated/table_natural_cases.tex}",
        r"\input{generated/table_claim_scope.tex}",
        "",
    ]
    wrapper_path.write_text("\n".join(wrapper_lines), encoding="utf-8")
    outputs[relative(wrapper_path)] = sha256(wrapper_path)

    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "evidence_tier": "confirmatory",
        "pilot_inputs_forbidden": True,
        "generator": relative(Path(__file__)),
        "generator_sha256": sha256(Path(__file__)),
        "paper_claims_sha256": sha256(CLAIMS),
        "source_sha256": {
            relative(path): sha256(path) for path in sorted(sources)
        },
        "table_sha256": dict(sorted(outputs.items())),
        "table_count": 7,
        "wrapper": relative(wrapper_path),
        "scope_policy": {
            "binary_claims": ["simple_vs_structured"],
            "controlled_descriptive_only": [
                "m03_profile", "m08_profile", "m09_counterexample",
                "detectability_exploitability_relation", "D_METHOD_CONDITIONAL",
                "model_family", "strength_response",
            ],
            "case_study_only": ["natural_cases"],
            "bootstrap_tail_p_values_labeled_inferential": False,
        },
    }
    manifest_path = OUTPUT / "result_tables_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    claims = validate_claims()
    if args.check_only:
        hashes = provenance(claims)
        task_registry()
        mechanism_profiles(hashes)
        mechanism_models(hashes)
        diagnostic_methods(hashes)
        strength_response(hashes)
        natural_cases()
        claim_scope(claims)
        print(json.dumps({"status": "FINAL_TABLE_INPUTS_PASS"}, sort_keys=True))
        return 0
    manifest = write_tables(claims)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
