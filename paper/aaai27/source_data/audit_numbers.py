#!/usr/bin/env python3
"""Compatibility entry point for the fail-closed corrected-v2 number audit."""

from generate_result_macros import main


if __name__ == "__main__":
    raise SystemExit(main(["--check-only"]))
