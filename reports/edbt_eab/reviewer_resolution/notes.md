# Notes: CDXR reviewer-risk resolution

## Frozen claim boundaries

- Primary positive claim: matched encoded-column cost only.
- Semantic-group recomposed overall interval crosses zero.
- Natural evidence covers five selected cases and is mixed.
- RF/LightGBM are a post-LR cross-learner extension whose design was frozen
  before their outcomes; their runner re-fitted strict/full baselines under a
  disclosed post-run deviation.

## Diagnostic targets

- Sparse: task and mechanism effects, recall, retention, and removal of the
  three construction-defined legitimate signal fields `x_000`, `x_003`, and
  `x_005`.
- NYC311: selection mask, selected field names, leak labels, recall, retention,
  repair opportunity, and repair advantage.

## Wording discipline

- Sparse signal concentration is a construction-grounded explanation candidate,
  not a separately randomized causal mechanism.
- NYC311 is a fixed-case failure diagnosis, not evidence about natural-task prevalence.
- Cross-learner agreement is corroborative and does not establish equivalence or invariance.

## Verified failure anatomy

- Sparse P3 selections: 1,100/1,100 hashes match the final B1 ledger.
- Sparse effect: -0.118332; all 4 tasks and 9/11 mechanism means are negative.
- Sparse localization: recall 0.569962, retention 0.848104. Other archetypes
  span recall 0.595645--0.627652 and retention 0.851077--0.853746.
- The sparse generator's legitimate signal fields are selected at rates
  x_000=0.892727, x_003=0.849091, and x_005=0.176364; an average of 1.918182
  of the three fields is removed per key.
- NYC311: all 3 P3 ledger hashes match. At k=8/40, opportunity is 0.019046,
  repair advantage is -0.108190, recall is 0.5, and retention is 0.815789.
  P3 removes `resolution_description`, misses `status`, and also removes seven
  contract-valid fields.
- These diagnostics required zero downstream model fits and are stored under
  `results/edbt_eab_revision/failure_anatomy/` with an independent manifest.
