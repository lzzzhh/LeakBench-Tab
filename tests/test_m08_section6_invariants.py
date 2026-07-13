"""Section-6 invariant tests for the structured_prior_v1 M08 amendment.

These complement tests/test_structured_prior_v1.py (self/same-timestamp label
exclusion, constant-prior fallback, determinism) with the remaining directive
invariants: strength effect, mask/field correspondence, unseen-entity prior,
and protocol-substitution prohibitions enforced by the frozen inference config.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.leakbench.mechanisms import MechanismConfig, MechanismID
from src.leakbench.mechanisms.structured_prior_v1 import (
    CONSTANT_PRIOR,
    StructuredPriorV1Injector,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/structured_prior_v1/inference_protocol_v1.json"


def _inject(strength, *, seed=17, noise_std=0.0, n=24):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = rng.randint(0, 2, size=n).astype(np.float32)
    timestamps = np.arange(n, dtype=float)
    entity_ids = np.repeat(np.arange(4), n // 4)
    config = MechanismConfig(
        MechanismID.ENTITY_LEAK,
        strength=strength,
        noise_std=noise_std,
        time_offset=0.0,
        prior_strength=2.0,
        seed=seed,
    )
    return StructuredPriorV1Injector(seed=seed).inject(
        X, y, [config], timestamps=timestamps, entity_ids=entity_ids,
        split_type="time",
    )


def test_m08_strength_zero_collapses_signal_to_constant_prior():
    """strength=0 => signal = prior everywhere (noiseless), independent of y."""
    task = _inject(strength=0.0, noise_std=0.0)
    feature = task.X[:, ~task.legitimate_mask].ravel()
    assert np.allclose(feature, CONSTANT_PRIOR)


def test_m08_strength_scales_deviation_from_prior_deterministically():
    """Higher strength linearly amplifies the deviation of the entity rate."""
    weak = _inject(strength=0.25, noise_std=0.0)
    strong = _inject(strength=1.0, noise_std=0.0)
    w = weak.X[:, ~weak.legitimate_mask].ravel() - CONSTANT_PRIOR
    s = strong.X[:, ~strong.legitimate_mask].ravel() - CONSTANT_PRIOR
    # rows with a non-trivial future rate must scale by exactly 4x (1.0/0.25)
    active = np.abs(s) > 1e-9
    assert active.any()
    ratio = s[active] / w[active]
    assert np.allclose(ratio, 4.0, atol=1e-6)


def test_m08_mask_marks_exactly_the_injected_field():
    task = _inject(strength=1.0)
    n_leak = int(task.leakage_mask.sum())
    assert n_leak == 1
    # the leaked column is the appended one; all original columns are legitimate
    n_orig = task.X.shape[1] - n_leak
    assert not task.leakage_mask[:n_orig].any()
    assert task.leakage_mask[-1]
    assert bool((~task.legitimate_mask == task.leakage_mask).all())


def test_m08_unseen_future_entity_row_uses_constant_prior():
    """The last row of every entity has no strict future -> constant prior."""
    task = _inject(strength=1.0, noise_std=0.0)
    rate = task.sample_metadata["m08_future_rate"]
    counts = task.sample_metadata["m08_future_count"]
    assert (rate[counts == 0] == CONSTANT_PRIOR).all()
    assert (counts == 0).any()


def test_m08_reproducible_by_seed():
    a = _inject(strength=1.0, noise_std=0.1, seed=7777)
    b = _inject(strength=1.0, noise_std=0.1, seed=7777)
    assert np.array_equal(a.X, b.X)


# ── Protocol-substitution prohibitions (frozen inference config) ──

def _protocol():
    return json.loads(PROTOCOL.read_text())


def test_protocol_forbids_mechanism_and_task_substitution():
    p = _protocol()
    ex = p["exclusions_and_missingness"]
    assert ex["mechanism_substitution"] == "forbidden"
    assert ex["task_substitution"] == "forbidden"
    assert ex["model_substitution"] == "forbidden"


def test_protocol_forbids_seed_substitution():
    assert _protocol()["exclusions_and_missingness"]["seed_substitution"] == "forbidden"


def test_protocol_metric_is_full_minus_strict_paired_harm():
    m = _protocol()["cell_metric"]
    assert m["name"] == "paired_harm"
    assert m["formula"] == "full_auc - strict_auc"
    assert m["strict_view"] == "task.X[:, ~task.leakage_mask]"
    assert m["full_view"] == "task.X"
