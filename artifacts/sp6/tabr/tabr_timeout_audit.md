# TabR Timeout Audit (SP6-G2)

4 of 5500 frozen-grid keys experienced a subprocess TimeoutExpired (1200s limit).
All resolved on retry with identical config (1800s limit after infrastructure fix).

No selection bias detected: timeouts span M01 (simple) and M05 (structured),
datasets 1 and 5, seeds 13/2026/7777 — no mechanism/dataset clustering.

The infrastructure timeout was increased from 1200s to 1800s between original and
retry runs. This is an **infrastructure amendment** (wall-clock scheduling), not a
model-protocol amendment. TabM training budget (max_epochs=100, patience=16) was
unchanged. See `tabr_timeout_audit.csv` for per-cell records.
