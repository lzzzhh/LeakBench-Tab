# EDBT EA&B paper workspace

This directory is the paper-facing EDBT 2027 Experiments, Analysis & Benchmarks
workspace and the repository's only active manuscript tree.

## Argument

CDXR is a contract-grounded evaluation architecture with a minimum information
contract, role-separated selector/evaluator/verifier interfaces, and a
machine-readable claim certificate. It separates construction validity (C),
blind detectability (D), learner exploitability (X), and matched-cost repair
response (R). LeakBench-Tab is its controlled instantiation. The frozen evidence
shows a positive primary encoded-cost repair advantage for LR, with
corroborative RF and LightGBM extensions under disclosed implementation and
baseline-refit deviations. Opportunity, archetype, semantic cost, and fixed
natural cases expose the boundary of that result.

## Build inputs

All paper figures, tables, and numeric macros use these three governed CSV files:

- `source_data/generated/main_results.csv`
- `source_data/generated/governance_results.csv`
- `source_data/generated/natural_cases.csv`

The asset builder verifies corrected-v2 measurement claims and the final EDBT
governance-revision claim state, analysis summaries, natural cases, semantic
cost sensitivity, revision manifest, and the separately manifested post-hoc
sparse/NYC311 failure anatomy before emitting those CSVs. Post-hoc diagnostics
cannot promote the frozen claim state.

Regenerate macros, tables, and figures with:

```bash
python paper/edbt_eab/source_data/build_paper_assets.py
python paper/edbt_eab/source_data/generate_paper_artifacts.py
```

Compile the draft with:

```bash
cd paper/edbt_eab
tectonic -X compile main.tex --outdir output/official
```

The manuscript uses the official EDBT 2027 A4 `acmart` template and the exact
`[EA&B]` title prefix. The template files were downloaded from the official
EDBT/ICDT 2027 call on 2026-07-17. Their SHA-256 values are:

- `acmart.cls`: `f5e24d4b29d24ab1cbdef46002febe7c2521f69b41c03d20bc92d8eef8fb5038`
- `ACM-Reference-Format.bst`: `a9f66287ef90d08b22344a20963b170faec86e26df1553f50b05cbb90fb545f3`
- `edbt-macros.tex`: `3bbc3404ecb535d0a0ba9cd534508d09ac5d252d66066a2b2c1c639c0a208843`

Tectonic's XeTeX path also requires `libertinusmath-regular.otf`. The local copy
is from Libertinus v7.051 and is accompanied by `OFL-Libertinus.txt`; it is a
local build dependency, not an EDBT template file. A conventional pdfLaTeX or
Overleaf build can use the official template without this extra local font.

Build the explicitly non-final EDBT artifact package with:

```bash
python paper/edbt_eab/source_data/build_edbt_draft_package.py
```

Build the compact, standalone LaTeX source package with:

```bash
python paper/edbt_eab/source_data/build_edbt_source_package.py
```

Build the complete paper bundle, including all figure formats, paper-facing
assets, generation scripts, compiled PDF, and compact evidence tables, with:

```bash
python paper/edbt_eab/source_data/build_edbt_complete_package.py
```

## Locked terminology

| Canonical term | Meaning |
|---|---|
| controlled panel task | One designed registry unit; do not call it a real dataset |
| construction validity (C) | Semantic invalidity under the prediction boundary |
| statistical detectability (D) | Diagnostic-conditional localization quality |
| model exploitability (X) | Permissive-minus-strict paired AUROC harm |
| repair opportunity (O) | Absolute strict--full AUROC distance available to repair |
| repair response (R) | SDR, contamination recall, and legitimate retention after intervention |
| repair advantage ($\Delta_R$) | P3 SDR minus within-key mean P2 SDR under matched cost |
| claim certificate ($\Gamma$) | Scope, cost, effect, interval, localization, provenance, and status |
| corroborative extension | Post-LR RF/LightGBM evidence frozen before its own outcomes; not original-protocol continuity |
| strict-distance reduction (SDR) | Movement of governed AUROC toward the strict reference |
| P2 matched random removal | Random policy removing the same number of fields as P3 |
| P3 blind MI removal | Training-side mutual-information ranking without mask access |
| bootstrap superiority probability | Fraction of bootstrap differences above zero; not a p-value |
| no reliable advantage | Interval crosses zero; not an equivalence claim |

## Figure palette

The selected palette comes from the provided ML visualization reference:

- P3: `#3F2B96`
- P2: `#A8C0FF`
- simple: `#FE9696`
- boundary: `#E6B94E`
- structured: `#B6D7A8`

The architecture figure is deliberately simple and remains replaceable by a
future author-designed figure.
