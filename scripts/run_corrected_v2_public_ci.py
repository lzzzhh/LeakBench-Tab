#!/usr/bin/env python3
"""Run the complete public-artifact verifier and public-only test profile."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    verifier = artifact / "scripts/verify_corrected_v2_public_artifact.py"
    if not verifier.is_file():
        raise FileNotFoundError(verifier)
    completed = subprocess.run(
        [sys.executable, str(verifier), str(artifact), "--run-tests"],
        cwd=artifact,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
