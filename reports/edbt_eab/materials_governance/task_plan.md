# Task Plan: EDBT EA&B Project Material Governance

## Goal

Create one evidence-first navigation layer for the RiskAgent/LeakBench-Tab project without moving, duplicating, or weakening frozen artifacts.

## Phases

- [x] Phase 1: Identify and install the relevant Claude Scholar skills.
- [x] Phase 2: Inventory project, experiment, evidence, paper, and reproducibility materials.
- [x] Phase 3: Classify canonical, derived, superseded, local-only, and working artifacts.
- [x] Phase 4: Build the canonical project-material index and evidence-to-paper map.
- [x] Phase 5: Validate paths, hashes, paper inputs, and repository state.

## Key Questions

1. Which files are authoritative for claims, tables, figures, and reproducibility?
2. Which materials are derived, superseded, local-only, or merely historical?
3. What is the smallest stable set a paper author, reviewer, or artifact evaluator needs?

## Decisions Made

- Use the Galaxy-Dawn Claude Scholar Codex branch because it supports project-wide research workflows.
- Install only planning, analysis, reporting, self-review, and ML-paper-writing skills; do not install hooks, agents, Zotero, Obsidian, or Git automation.
- Add a navigation layer instead of moving frozen assets, because paths and hashes are part of the provenance contract.

## Errors Encountered

- `generate_paper_artifacts.py --help` regenerated deterministic artifacts
  because the script has no argument parser. No evidence input changed; the
  behavior is now documented in the material index.

## Status

**Complete** - canonical navigation, asset dispositions, release blockers, and
verification results are recorded in `reports/edbt_eab/PROJECT_MATERIALS.md`.
