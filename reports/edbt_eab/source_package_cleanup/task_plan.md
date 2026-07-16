# Task Plan: Remove AAAI Paper Assets and Build EDBT Source Package

## Goal

Remove AAAI-specific paper and release materials while preserving scientific
evidence, then deliver a self-contained EDBT LaTeX source ZIP that compiles after
extraction.

## Phases

- [x] Phase 1: Inventory every AAAI path and all live references to it.
- [x] Phase 2: Define deletion/preservation boundary and update EDBT tooling.
- [x] Phase 3: Delete AAAI-only assets and build the standalone source package.
- [x] Phase 4: Extract, compile, render-check, test, and document the result.

## Success Criteria

1. No tracked or untracked `paper/aaai27` tree or AAAI-named release package remains.
2. No active EDBT documentation, builder, or test depends on an AAAI paper path.
3. The source ZIP contains `main.tex`, bibliography, official template files,
   generated tables, figures, and local build dependencies.
4. A clean extraction compiles to an A4 PDF without repository-relative inputs.

## Decisions Made

- Preserve corrected-v2 results and frozen protocols even if historical metadata
  mentions AAAI; those are scientific provenance, not AAAI paper assets.
- Remove venue-specific paper sources, builders, tests, and release packages.
- Keep the compact EDBT evidence artifact separate from the requested LaTeX
  source package; the source ZIP will contain only compilation inputs.

## Errors Encountered

- The first clean-extraction command selected a working directory that did not
  exist until the same command created it, so process startup failed. Extraction
  and compilation were rerun as two commands and succeeded.

## Status

**Complete** - EDBT-only repository paper boundary and verified source ZIP delivered.
