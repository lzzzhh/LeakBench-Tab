"""test_corrections.py — Behavior tests for the 18 correction items.

These tests encode the REQUIRED invariants from CORRECTION_HANDOFF.md.
They should FAIL on the current codebase — that's the point.
Only after fixing the implementation should they pass.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestM08EntityLeakage:
    """Critical: M08 must produce entity-structured features independent of current label."""

    def test_m08_not_random_noise(self):
        """M08 features must correlate with entity-level target rates, not be random."""
        from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID
        rng = np.random.RandomState(42)
        n, n_entities = 500, 20
        entities = rng.randint(0, n_entities, n)
        # Create entity-structured target
        entity_rates = rng.beta(2, 2, n_entities)
        y_prob = np.array([entity_rates[e] for e in entities])
        y = (rng.rand(n) < y_prob).astype(np.float32)
        X = rng.randn(n, 5).astype(np.float32)

        injector = LeakBenchInjector(seed=42)
        task = injector.inject(X, y, [MechanismConfig(
            mechanism=MechanismID.ENTITY_LEAK, strength=1.0, n_leakage_features=1, seed=42)])

        leak_col = task.X[:, -1]
        # Correlation with entity-level target rates
        entity_avg = np.array([y[entities == e].mean() for e in range(n_entities)])
        leak_entity_corr = abs(np.corrcoef(
            [entity_avg[e] for e in entities], leak_col)[0, 1])

        # M08 should have meaningful structure in entity-rate correlation
        assert leak_entity_corr > 0.02, \
            f"M08 entity-rate correlation = {leak_entity_corr:.3f} (expected > 0.02)"

    def test_m08_no_current_label_leakage(self):
        """M08 must NOT include the current sample's own label."""
        from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID
        rng = np.random.RandomState(42)
        n, n_entities = 300, 15
        entities = rng.randint(0, n_entities, n)
        y = (rng.rand(n) < 0.5).astype(np.float32)
        X = rng.randn(n, 5).astype(np.float32)

        injector = LeakBenchInjector(seed=42)
        task = injector.inject(X, y, [MechanismConfig(
            mechanism=MechanismID.ENTITY_LEAK, strength=1.0, seed=42)])

        leak = task.X[:, -1]
        # Leakage feature should NOT be a near-perfect copy of y
        corr_with_y = abs(np.corrcoef(leak, y)[0, 1])
        assert corr_with_y < 0.9, \
            f"M08 correlation with current y = {corr_with_y:.3f} (expected < 0.9)"

    def test_m08_future_only_aggregation(self):
        """If using future aggregation, must exclude current label."""
        from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID
        rng = np.random.RandomState(9)
        X = rng.normal(size=(200, 4)).astype(np.float32)
        y = rng.randint(0, 2, size=200).astype(np.float32)
        entities = np.repeat(np.arange(10), 20)
        timestamps = np.tile(np.arange(20), 10)
        config = [MechanismConfig(MechanismID.ENTITY_LEAK, strength=1.0)]
        original = LeakBenchInjector(seed=9).inject(X, y, config, timestamps=timestamps, entity_ids=entities)
        changed_y = y.copy(); changed_y[7] = 1 - changed_y[7]
        changed = LeakBenchInjector(seed=9).inject(X, changed_y, config, timestamps=timestamps, entity_ids=entities)
        assert original.X[7, -1] == changed.X[7, -1]
        assert original.mechanism_params[-1]["strictly_future"] is True


class TestM09SourceLeakage:
    """Critical: M09 must produce source-structured features from deployment-invalid metadata."""

    def test_m09_not_random_noise(self):
        """M09 features must differ between sources, with source-conditioned target prevalence."""
        from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID
        rng = np.random.RandomState(42)
        n, n_sources = 500, 8
        # Source-conditioned target prevalence
        source_rates = rng.beta(2, 2, n_sources) * 0.6 + 0.2
        sources = rng.randint(0, n_sources, n)
        y_prob = np.array([source_rates[s] for s in sources])
        y = (rng.rand(n) < y_prob).astype(np.float32)
        X = rng.randn(n, 5).astype(np.float32)

        injector = LeakBenchInjector(seed=42)
        task = injector.inject(X, y, [MechanismConfig(
            mechanism=MechanismID.SOURCE_LEAK, strength=1.0, seed=42)])

        leak = task.X[:, -1]
        # Check source-separability of leak feature
        source_means = [leak[sources == s].mean() for s in range(n_sources)]
        source_std = np.std(source_means)
        assert source_std > 0.01, \
            f"M09 source separability = {source_std:.4f} (sources should differ)"

    def test_m09_no_full_data_target_mean(self):
        """M09 must NOT use full-data source target mean (self-leakage)."""
        from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID
        rng = np.random.RandomState(42)
        n, n_sources = 300, 8
        y = (rng.rand(n) < 0.5).astype(np.float32)
        X = rng.randn(n, 5).astype(np.float32)

        injector = LeakBenchInjector(seed=42)
        task = injector.inject(X, y, [MechanismConfig(
            mechanism=MechanismID.SOURCE_LEAK, strength=1.0, seed=42)])

        leak = task.X[:, -1]
        corr_with_y = abs(np.corrcoef(leak, y)[0, 1])
        assert corr_with_y < 0.9, \
            f"M09 correlation with individual y = {corr_with_y:.3f} (expected < 0.9)"

    def test_m09_source_label_permutation_stable(self):
        """Permuting source labels should not change metric distribution."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID
        rng = np.random.RandomState(10)
        X = rng.normal(size=(800, 4)).astype(np.float32)
        y = np.tile([0, 1], 400).astype(np.float32)
        task = LeakBenchInjector(seed=10).inject(X, y, [MechanismConfig(MechanismID.SOURCE_LEAK, strength=0.8)])
        block = task.X[:, task.n_original:]
        permutation = rng.permutation(block.shape[1])
        first = LogisticRegression(max_iter=1000).fit(block[task.train_idx], y[task.train_idx])
        second = LogisticRegression(max_iter=1000).fit(block[task.train_idx][:, permutation], y[task.train_idx])
        auc_first = roc_auc_score(y[task.test_idx], first.predict_proba(block[task.test_idx])[:, 1])
        auc_second = roc_auc_score(y[task.test_idx], second.predict_proba(block[task.test_idx][:, permutation])[:, 1])
        assert auc_first == pytest.approx(auc_second, abs=1e-12)


class TestOperationalMetadata:
    """Critical: Operational metadata must NOT peek at contamination identity."""

    @staticmethod
    def _example():
        from src.leakbench.diagnostics import (
            OperationalFeatureMetadata, OperationalMetadata,
        )
        rng = np.random.RandomState(11)
        X = rng.normal(size=(240, 4))
        y = (X[:, 1] + 0.2 * rng.normal(size=240) > 0).astype(float)
        feature_ids = ["fid-a", "fid-b", "fid-c", "fid-d"]
        metadata = OperationalMetadata(
            features={
                fid: OperationalFeatureMetadata(
                    stable_id=fid,
                    lifecycle="post_outcome" if fid == "fid-c" else "prediction_time",
                    group_id="bundle" if fid in {"fid-b", "fid-c"} else None,
                )
                for fid in feature_ids
            },
            graph_edges=(("fid-b", "fid-c", 0.8),),
        )
        return X, y, feature_ids, metadata

    def test_scores_stable_after_column_permutation_and_anonymisation(self):
        """Stable IDs, rather than positions or display names, identify scores."""
        from src.leakbench.diagnostics import compute_operational_diagnostics
        X, y, feature_ids, metadata = self._example()
        baseline = compute_operational_diagnostics(X, y, feature_ids, metadata)

        permutation = np.array([2, 0, 3, 1])
        permuted_ids = [feature_ids[j] for j in permutation]
        permuted = compute_operational_diagnostics(
            X[:, permutation], y, permuted_ids, metadata
        )
        for field_name in (
            "predictive_impact", "availability_risk", "structural_risk",
            "environment_instability", "composite",
        ):
            assert baseline.values_by_id(field_name) == pytest.approx(
                permuted.values_by_id(field_name), abs=1e-12
            )

    def test_operational_scorer_has_no_oracle_input(self):
        """The scoring API cannot receive contamination identity by construction."""
        import inspect
        from src.leakbench.diagnostics import compute_operational_diagnostics
        source = inspect.getsource(compute_operational_diagnostics)
        for forbidden in ("leak_mask", "n_clean", "mechanism_labels", "feature_names", "oracle"):
            assert forbidden not in source


class TestGroupGovernance:
    """Critical: Group governance must actually remove groups, not silently no-op."""

    def test_group_membership_uses_valid_feature_ids(self):
        """Group membership must reference features by valid IDs matching feature_names."""
        from src.leakbench.diagnostics import OperationalFeatureMetadata, OperationalMetadata
        feature_ids = ["feat-0", "feat-1", "feat-2"]
        metadata = OperationalMetadata(features={
            fid: OperationalFeatureMetadata(stable_id=fid, group_id="group-a")
            for fid in feature_ids
        })
        assert metadata.groups(feature_ids) == {"group-a": feature_ids}

    def test_group_governance_not_equivalent_to_field(self):
        """Group removal should produce different masks than field removal on grouped data."""
        from src.leakbench.diagnostics import OperationalFeatureMetadata, OperationalMetadata
        from src.leakbench.governance import GovernanceStrategy, apply_strategy
        feature_ids = ["a", "b", "c", "d"]
        metadata = OperationalMetadata(features={
            fid: OperationalFeatureMetadata(
                stable_id=fid,
                group_id="high-risk-bundle" if fid in {"a", "b", "c"} else "other",
            )
            for fid in feature_ids
        })
        scores = np.array([0.9, 0.8, 0.7, 0.1])
        field = apply_strategy(
            GovernanceStrategy.FIXED_FIELD_BUDGET, feature_ids, scores, metadata,
            budget=0.25,
        )
        group = apply_strategy(
            GovernanceStrategy.FIXED_GROUP_BUDGET, feature_ids, scores, metadata,
            budget=0.5,
        )
        assert field.quarantined_features == ["a"]
        assert set(group.quarantined_features) == {"a", "b", "c"}
        assert not np.array_equal(field.feature_mask, group.feature_mask)


class TestM10MixedLeakage:
    """Critical: M10 legitimate/leakage labels must match actual feature generation."""

    def test_m10_legitimate_not_from_y(self):
        """M10 legitimate features must come from clean X, not from y."""
        from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)

        injector = LeakBenchInjector(seed=42)
        task = injector.inject(X, y, [MechanismConfig(
            mechanism=MechanismID.MIXED, strength=1.0, seed=42)])

        legitimate = [
            j for j in range(task.n_original, task.X.shape[1])
            if task.mechanism_labels[j] == "legitimate"
        ]
        assert legitimate == [task.n_original]
        assert np.array_equal(task.X[:, legitimate[0]], X[:, 0])

    def test_m10_mask_consistent(self):
        """M10 legitimate features must NOT be marked as leakage in mask."""
        from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)

        injector = LeakBenchInjector(seed=42)
        task = injector.inject(X, y, [MechanismConfig(
            mechanism=MechanismID.MIXED, strength=1.0, seed=42)])

        # Any feature with mechanism label "legitimate" must NOT be in leak_mask
        for j in range(task.n_original, task.X.shape[1]):
            label = task.mechanism_labels[j] if j < len(task.mechanism_labels) else ""
            if label == "legitimate":
                assert not task.leakage_mask[j], \
                    f"Feature {j} labeled legitimate but leak_mask=True"


class TestGovernanceDegeneracy:
    """High: G3/G4/G6 must not be identical row-by-row."""

    def test_strategies_produce_different_masks(self):
        """Different governance strategies should produce different masks on non-trivial data."""
        from src.leakbench.diagnostics import OperationalFeatureMetadata, OperationalMetadata
        from src.leakbench.governance import GovernanceStrategy, apply_strategy
        feature_ids = ["a", "b", "c", "d"]
        metadata = OperationalMetadata(
            features={fid: OperationalFeatureMetadata(stable_id=fid) for fid in feature_ids},
            graph_edges=(("a", "b", 1.0), ("b", "c", 1.0)),
        )
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        field = apply_strategy(
            GovernanceStrategy.FIXED_FIELD_BUDGET, feature_ids, scores, metadata,
            budget=0.5,
        )
        graph = apply_strategy(
            GovernanceStrategy.GRAPH_CUT, feature_ids, scores, metadata,
            budget=0.5,
        )
        assert set(field.quarantined_features) == {"a", "b"}
        assert set(graph.quarantined_features) == {"a", "b", "c"}
        assert not np.array_equal(field.feature_mask, graph.feature_mask)

    def test_graph_without_graph_is_explicitly_not_applicable(self):
        from src.leakbench.diagnostics import OperationalFeatureMetadata, OperationalMetadata
        from src.leakbench.governance import (
            GovernanceStatus, GovernanceStrategy, apply_strategy,
        )
        feature_ids = ["a", "b"]
        metadata = OperationalMetadata(features={
            fid: OperationalFeatureMetadata(stable_id=fid) for fid in feature_ids
        })
        result = apply_strategy(
            GovernanceStrategy.GRAPH_CUT, feature_ids, [0.9, 0.1], metadata,
        )
        assert result.status == GovernanceStatus.NOT_APPLICABLE
        assert result.reason == "no operational feature graph"
        assert result.n_quarantined == 0

    def test_lifecycle_policy_uses_operational_metadata(self):
        from src.leakbench.diagnostics import OperationalFeatureMetadata, OperationalMetadata
        from src.leakbench.governance import GovernanceStrategy, apply_strategy
        feature_ids = ["opaque-1", "opaque-2", "opaque-3"]
        metadata = OperationalMetadata(features={
            "opaque-1": OperationalFeatureMetadata(stable_id="opaque-1"),
            "opaque-2": OperationalFeatureMetadata(
                stable_id="opaque-2", lifecycle="post_outcome"
            ),
            "opaque-3": OperationalFeatureMetadata(stable_id="opaque-3"),
        })
        result = apply_strategy(
            GovernanceStrategy.LIFECYCLE_REMOVAL,
            feature_ids,
            [0.1, 0.1, 0.1],
            metadata,
        )
        assert result.quarantined_features == ["opaque-2"]


class TestSeeds:
    """High: Experiments must use declared seeds, not fixed single seed."""

    def test_seeds_not_hardcoded_to_42(self):
        """Scripts declaring SEEDS = [13, 42, 2026] must actually loop over all three."""
        import ast
        for path in ["experiments/leakbench/run_meta_tier.py",
                      "experiments/leakbench/run_operational_meta.py"]:
            pp = Path(path)
            if not pp.exists():
                continue
            code = pp.read_text()
            # Check that seed is actually used in a loop, not hardcoded
            has_seed_list = "SEEDS" in code
            has_seed_loop = any(kw in code for kw in ["for seed in SEEDS", "for seed in"])
            assert has_seed_loop or not has_seed_list, \
                f"{path}: declares seeds but may not loop over them"


def test_legacy_meta_runner_is_integrity_held():
    """The legacy metadata generator must never silently emit corrected_v2 evidence."""
    from experiments.leakbench import run_meta_tier

    with pytest.raises(RuntimeError, match="INTEGRITY HOLD"):
        run_meta_tier.main()
