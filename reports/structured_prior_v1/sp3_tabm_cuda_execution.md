# Gate SP3 — TabM CUDA Execution (Option D: WSL2 + CUDA)

**Status: COMPLETE. 1500/1500 SUCCESS on CUDA, no fallback.**
**Date:** 2026-07-13

## Decision path

Option D was chosen over A/B/C: run the **unmodified byte-frozen runner** on the
Windows RTX 4060 node via **WSL2 Ubuntu + CUDA**, so the frozen `device: cuda`
config is satisfied without editing any hash-protected file.

## Why the earlier Windows-native attempt failed

`run_structured_prior_v1_bundle.py`'s `_relative()` yields backslash paths on
native Windows (`configs\paper\...`) that don't match the forward-slash keys in
the freeze manifest, so the runner's own integrity check raised
"file differs from protocol freeze". The runner is byte-frozen and self-verifies,
so it cannot be patched. WSL2 provides a Linux (posix) filesystem where the
unmodified runner passes its own checks — identical to the macOS CPU environment.

## Setup (all requirements met)

| Requirement | Status |
|-------------|--------|
| No modification of frozen runner/config/manifest/bundle | ✓ (14/14 hashes verified in WSL) |
| Project in Linux FS `~/leakbench-tab`, not `/mnt/c` | ✓ (`/root/leakbench-tab`, ext4) |
| `git config core.autocrlf false` | ✓ |
| Frozen hash re-check before run | ✓ 14/14 in WSL |
| `nvidia-smi` in WSL | ✓ RTX 4060, driver 560.94, CUDA 12.6 |
| `torch.cuda.is_available()` | ✓ True |
| PyTorch recognizes RTX 4060 | ✓ "NVIDIA GeForce RTX 4060 Laptop GPU" |
| device = cuda (no CPU/MPS fallback) | ✓ model_manifest device=cuda for all 1500 |

## Smoke test (excluded)

1 cell (M08/S3/seed13/panel_00) → `_excluded_smoke/tabm_smoke_EXCLUDED.csv`
(temp dir, not in frozen results). SUCCESS, integrity_verified, device=cuda,
implementation `tabm.TabM@0.0.3`. Verified runner path check, CUDA call, and
result persistence before the full run.

## Full run

| Metric | Value |
|--------|-------|
| Cells | 1500 / 1500 |
| SUCCESS | 1500 |
| FAILURE | 0 |
| integrity_verified | 1500 |
| duplicates (task_hash, model) | 0 |
| non-finite AUC | 0 |
| devices used | {cuda} only |
| Coverage | M04/M05/M08 × S1–S5 × 5 seeds × 20 datasets = 100/mech×strength |
| Runtime | ~5886s (~98 min) incl. one resume |
| Ledger SHA256 | `ea04c3e822df1f1e…` (verified identical after transfer to Mac) |

Note: the run was launched via a Windows scheduled task so the WSL distro stays
alive independent of the SSH session (WSL2 tears down the distro when the last
attached process exits). The frozen runner's `--resume` re-verified all hashes on
restart; the first 10 pre-resume cells were re-validated, not double-counted
(dedup on `run_id`).

## Execution environment (recorded, not a protocol deviation)

Windows host LAPTOP-5HGON8HL → WSL2 Ubuntu (kernel 6.18-microsoft-standard-WSL2)
→ RTX 4060 Laptop GPU, driver 560.94, torch 2.5.1+cu121 (CUDA 12.1 runtime),
Python 3.12.3, tabm 0.0.3. Full record:
`results/structured_prior_replacement_v1/tabm_execution_environment.json`.
