#!/usr/bin/env python3
"""T0-B1R2.2 Environment Capture."""
import json, os, platform, sys, subprocess
from pathlib import Path
import numpy, pandas, sklearn, scipy

ROOT = Path(__file__).resolve().parents[2]
head = subprocess.run(["git","rev-parse","HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()

env = {
    "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
    "numpy": numpy.__version__, "pandas": pandas.__version__,
    "sklearn": sklearn.__version__, "scipy": scipy.__version__,
    "cpu_count_logical": os.cpu_count(),
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
    "git_sha": head,
    "scientific_freeze_sha": "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845",
    "r2_runner_seal_sha": "b1fb0041a2e6ebdef817b5be489b9c85993002de",
    "validation_scope": "LOCAL_VALIDATION_ONLY",
}
out = ROOT / "results/edbt_t0_b_dryrun_r2_3"; out.mkdir(parents=True, exist_ok=True)
with open(out / "environment_receipt_r2_3.json", "w") as f: json.dump(env, f, indent=2)
print("Environment captured")
