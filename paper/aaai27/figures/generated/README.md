# Required final data figures

The main paper expects exactly these final-only PDF assets:

- `cdx_scatter.pdf`: mechanism-level primary-MI D versus paired X, with
  hierarchical intervals, category encoding, and M03/M08/M09 labels.
- `mechanism_model_heatmap.pdf`: paired harm by all eleven mechanisms and five
  core model families, with one common color scale and explicit failed-cell
  handling.
- `strength_diagnostic_robustness.pdf`: strength response plus four oracle-blind,
  pre-test development rankers (MI primary, absolute correlation, and LR
  coefficient are training-only; RF permutation is evaluated on frozen
  validation labels).

No generation script is supplied by the paper scaffold.  These files must be
generated from the final canonical/statistics artifacts, never from pilot
tables.  Once `paper_claims.json` passes and results macros are active, LaTeX
raises an error if any required figure is missing.
