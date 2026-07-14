# Legacy Model Implementation Audit (SP6-A0)

All four prior modern-model implementations are marked **IMPLEMENTATION_UNVERIFIED / NON_CLAIM_ELIGIBLE / SUPERSEDED_FOR_SP6**.
Usable only for bug-tracing / config comparison, never for formal results.

## ModernNCA
- Old: `src/leakbench/models/modern_models.py::train_evaluate_modernnca`
- Source: sklearn NeighborhoodComponentsAnalysis + KNeighborsClassifier (NOT the 2024 deep ModernNCA)
- Bug: impersonates deep metric-learning model with classical NCA+kNN; falls back to plain kNN
- Exclusion: IMPLEMENTATION_UNVERIFIED / not official implementation

## TabR
- Old: `src/leakbench/models/modern_models.py::train_evaluate_tabr`
- Source: KNeighborsRegressor + variance-scaled features (self-described 'Simplified')
- Bug: comment states 'Full TabR would use a learned encoder'; this is a kNN baseline
- Exclusion: IMPLEMENTATION_UNVERIFIED / simplified kNN, not TabR

## TabPFNv2
- Old: `src/leakbench/models/modern_models.py::train_evaluate_tabpfn`
- Source: official tabpfn package call (device=cpu)
- Bug: 594/594 cells returned constant 0.000 (API failures)
- Exclusion: RUNTIME_FAILURE / constant-zero output

## TabICL
- Old: `NONE`
- Source: not present in repository
- Bug: no implementation exists
- Exclusion: NOT_IMPLEMENTED

