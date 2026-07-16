# Task Plan: EDBT EA&B Pre-Submission Review

## Goal

Produce an evidence-grounded submission-readiness verdict, fix safe local issues,
and create a correctly labeled EDBT draft artifact package without weakening any
frozen claim or protocol.

## Phases

- [x] Phase 1: Audit manuscript structure, claims, citations, figures, tables, and front matter.
- [x] Phase 2: Audit release builder and current package contents against the EDBT paper boundary.
- [x] Phase 3: Apply minimal safe fixes and create the EDBT draft package.
- [x] Phase 4: Compile, render, test, verify hashes, and write the final review report.

## Key Questions

1. Does every main manuscript claim stay within `paper_claims.json` or the SP8 claim matrix?
2. Are all citations resolvable and used for claims they can support?
3. Does the draft package contain EDBT paper sources and exclude stale, pilot, superseded, and local-only materials?
4. Which remaining blockers require author or venue input?

## Decisions Made

- Operate in camera-ready audit mode, but keep the package labeled `draft` until author and venue metadata are confirmed.
- Do not modify frozen result, claim, protocol, or governance files.
- Preserve the existing mislabeled ZIP for provenance; create a new correctly labeled package instead of overwriting it.

## Errors Encountered

- The first official-template Tectonic build failed because the XeTeX branch of
  `acmart` could not find `libertinusmath-regular.otf`. Resolved with the
  upstream Libertinus v7.051 font and OFL license for local builds.
- One combined verification command used a repository-relative path while its
  working directory was already `paper/edbt_eab/`. Reran from the repository
  root successfully.
- Unscoped `pytest -q` initially collected duplicate test modules from the
  historical unpacked release directory. `pytest.ini` now confines discovery
  to the authoritative `tests/` tree; both scoped and root commands pass.

## Status

**Complete** - conditional submission verdict recorded in `PRESUBMISSION_REVIEW.md`.
