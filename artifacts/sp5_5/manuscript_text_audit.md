# Manuscript Text Audit (SP5.5)

Files searched: 4 (main.tex, supplement.tex, generated + base macros)
Raw matches: 4
COINCIDENTAL: 4 (0.05 in strength set / learning rate / p-value)
VALID_CURRENT: 0
**SUPERSEDED_ACTIVE: 0** (requirement: 0)

## Detail
- [COINCIDENTAL] supplement.tex:91 `old_structured_auprc_005` :: $\{0.05,0.10,0.20,0.35,0.50\}$, and M11 component sizes
- [COINCIDENTAL] supplement.tex:150 `old_structured_auprc_005` :: CatBoost & None & 250 iterations, rate 0.05, depth 6, CPU & Validation AUC; patience 30 \\
- [COINCIDENTAL] supplement.tex:151 `old_structured_auprc_005` :: LightGBM & None & 250 trees, rate 0.05, 31 leaves, depth 6, min child 20 &
- [COINCIDENTAL] result_macros.tex:21 `old_structured_auprc_005` :: \renewcommand{\LBIncrementalPermutationP}{<0.05}

## Conclusion
No superseded headline reference (old structured AUPRC 0.05-0.07, M08 0.048, ratios 2.2x/2.5x, r=0.73, within-category-zero, category-driven, fully-consistent, eight-models, undetectable) appears in the active manuscript. All 0.05 matches are coincidental (contamination strength grid, learning rate, p-value threshold).
