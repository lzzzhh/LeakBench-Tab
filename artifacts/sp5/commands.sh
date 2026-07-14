#!/bin/bash
# SP5-G reproducible pipeline
python3 scripts/assemble_claim_ledger_inputs_v2.py
python3 scripts/compute_sp4_detectability.py
python3 scripts/build_claim_ledger_v2.py
python3 scripts/recompute_sp5_claims.py
python3 scripts/render_sp5_figures.py
python3 -m pytest tests/test_sp5_ledger.py -q
