# Related-work and novelty audit

## What is not new

- Prediction-time legitimacy and the no-time-machine principle are established by Kaufman, Rosset, and Perlich (KDD 2011).
- Leakage taxonomies and their effect on reproducibility are established by Kapoor and Narayanan (Patterns 2023).
- Prediction cadence, perspective, and applicability have already been used to reason about label leakage in health-care prediction by Davis et al. (JAMIA 2024).
- Variable performance inflation across leakage forms has already been demonstrated in a controlled connectome study by Rosenblatt et al. (Nature Communications 2024); heterogeneity of leakage harm is therefore prior art, not our standalone novelty.
- Magar and Schwartz (ACL 2022) already distinguish contamination, memorization, and downstream exploitation; the exploitation concept is prior art even though their setting is NLP pretraining rather than prediction-time tabular features.
- Subotić, Bojanić, and Stojić (SOAP 2022) already detect leakage-prone data-science code statically; LeakBench-Tab is not the first leakage detector.
- Roth (arXiv:2604.04199, 2026) already presents a quantitative, large-scale comparison of leakage-type severity across tabular datasets.
- Large multi-dataset tabular comparisons already exist, including TableShift and TabZilla. Experiment count or inclusion of a neural tabular model is not a novelty claim.

## Defensible novelty target

The paper should claim a benchmark-level operationalization, not invention of leakage legitimacy:

> LeakBench-Tab jointly measures semantic contamination validity, feature-level statistical detectability, and paired model exploitability under construction-audited mechanisms and aligned strict/permissive protocols.

The empirical novelty is the crossed C/D/X profiles under a controlled mechanism matrix: an invalid field can be difficult to localize yet strongly exploited, or statistically localizable yet not improve the evaluated model. The category-level D--X association must be reported together with leave-one-mechanism-out and within-category sensitivity, rather than as a universal monotone law.

## Prohibited novelty wording

- Do not claim to be the first work to define data leakage, prediction-time validity, or the no-time-machine rule.
- Do not claim to introduce a new detector, governance algorithm, or tabular architecture.
- Do not use cell count as the primary novelty argument.
- Do not claim universal separation of simple and structured contamination; M09 is the declared structured counterexample.  Its eight-column one-hot D is representation-conditional, not a field-level detectability generality claim, and its designed-category reweighting interval is descriptive only.
- Do not use an unqualified "first" claim. The defensible distinction is the joint, mechanism-centric tabular measurement design, not priority over every domain-specific leakage audit.
- In particular, do not claim the first large-scale leakage study, the first quantitative leakage comparison, or the first leakage-type severity landscape.

## Main-track fit

The intended fit is empirical, integrative, and critical evaluation in the AAAI Main Technical Track. The benchmark should be presented as measurement infrastructure with auditable negative results, consistent with AAAI's stated acceptance of empirical and critical contributions. Metadata, governance, and natural transfer remain secondary unless corrected evidence independently clears their acceptance criteria.
