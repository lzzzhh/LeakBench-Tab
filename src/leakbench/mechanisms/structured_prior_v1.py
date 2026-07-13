"""Outcome-independent structured-mechanism amendment for M04/M05/M08.

The original injector is intentionally left unchanged.  This opt-in subclass
replaces every fallback/shrinkage prior used by the three future-outcome
mechanisms with the fixed, outcome-independent value 0.5.  A row may therefore
depend on strictly later labels, but never on its own label, labels at the same
timestamp, or any full-table label statistic.
"""
from __future__ import annotations

import numpy as np

from src.leakbench.mechanisms import LeakBenchInjector


CONSTANT_PRIOR = 0.5
AMENDMENT_VERSION = "structured_constant_prior_v1"


class StructuredPriorV1Injector(LeakBenchInjector):
    """Opt-in injector with a fixed 0.5 prior for M04, M05, and M08."""

    def _future_window(self, y, timestamps, skip, width):
        """Return strictly-future window means with a constant empty fallback.

        ``side="right"`` excludes the current row and every row sharing its
        timestamp.  ``skip`` retains the original M05 row-offset convention,
        but is applied only after locating the first strictly later timestamp.
        """
        y = np.asarray(y, dtype=float)
        timestamps = np.asarray(timestamps)
        order = np.argsort(timestamps, kind="stable")
        ordered_time = timestamps[order]
        ordered_y = y[order]
        cumulative = np.concatenate([[0.0], np.cumsum(ordered_y, dtype=float)])
        result = np.full(len(y), CONSTANT_PRIOR, dtype=float)
        counts = np.zeros(len(y), dtype=int)
        for row in range(len(y)):
            first_strict_future = np.searchsorted(
                ordered_time, timestamps[row], side="right"
            )
            start = first_strict_future + max(0, int(skip) - 1)
            end = min(len(y), start + int(width))
            if start < end:
                result[row] = (cumulative[end] - cumulative[start]) / (end - start)
                counts[row] = end - start
        return result, counts

    def _m04(self, c, X, y, timestamps, entity_ids, source_ids):
        width = max(2, int(len(y) * 0.05))
        future, counts = self._future_window(y, timestamps, skip=1, width=width)
        signal = CONSTANT_PRIOR + c.strength * (future - CONSTANT_PRIOR)
        signal += self.rng.normal(0.0, c.noise_std, len(y))
        return self._block(
            signal.reshape(-1, 1),
            ["contam_M04_0"],
            ["M04"],
            [{
                "strength": c.strength,
                "window": width,
                "excludes_current": True,
                "excludes_same_timestamp": True,
                "strictly_future": True,
                "prior_type": "outcome_independent_constant",
                "prior_value": CONSTANT_PRIOR,
                "amendment_version": AMENDMENT_VERSION,
            }],
            metadata={"m04_future_count": counts},
        )

    def _m05(self, c, X, y, timestamps, entity_ids, source_ids):
        skip = (
            max(1, int(len(y) * c.time_offset))
            if c.time_offset > 0
            else max(1, int(len(y) * 0.05))
        )
        width = max(2, skip)
        future, counts = self._future_window(y, timestamps, skip=skip, width=width)
        signal = CONSTANT_PRIOR + c.strength * (future - CONSTANT_PRIOR)
        signal += self.rng.normal(0.0, c.noise_std, len(y))
        return self._block(
            signal.reshape(-1, 1),
            ["contam_M05_0"],
            ["M05"],
            [{
                "strength": c.strength,
                "look_ahead_rows": skip,
                "window": width,
                "excludes_current": True,
                "excludes_same_timestamp": True,
                "strictly_future": True,
                "prior_type": "outcome_independent_constant",
                "prior_value": CONSTANT_PRIOR,
                "amendment_version": AMENDMENT_VERSION,
            }],
            metadata={"m05_future_count": counts},
        )

    def _m08(self, c, X, y, timestamps, entity_ids, source_ids):
        n = len(y)
        if entity_ids is None:
            n_entities = min(max(2, int(c.n_entities)), max(2, n // 20))
            entity_ids = np.arange(n) % n_entities
            self.rng.shuffle(entity_ids)
        else:
            entity_ids = np.asarray(entity_ids)
            n_entities = len(np.unique(entity_ids))

        future_rate = np.full(n, CONSTANT_PRIOR, dtype=float)
        future_count = np.zeros(n, dtype=int)
        prior_weight = max(0.0, float(c.prior_strength))
        for entity in np.unique(entity_ids):
            rows = np.flatnonzero(entity_ids == entity)
            ordered = rows[np.argsort(timestamps[rows], kind="stable")]
            ordered_time = timestamps[ordered]
            ordered_y = y[ordered]
            cumulative = np.concatenate([[0.0], np.cumsum(ordered_y, dtype=float)])
            for row in ordered:
                start = np.searchsorted(
                    ordered_time, timestamps[row], side="right"
                )
                count = len(ordered) - start
                future_count[row] = count
                if count:
                    total = cumulative[-1] - cumulative[start]
                    future_rate[row] = (
                        total + prior_weight * CONSTANT_PRIOR
                    ) / (count + prior_weight)

        strength = float(np.clip(c.strength, 0.0, 1.0))
        signal = CONSTANT_PRIOR + strength * (future_rate - CONSTANT_PRIOR)
        signal += self.rng.normal(0.0, c.noise_std, n)
        params = [{
            "strength": strength,
            "n_entities": int(n_entities),
            "aggregation": "strict_future_same_entity_constant_shrinkage_mean",
            "prior_weight": prior_weight,
            "prior_type": "outcome_independent_constant",
            "prior_value": CONSTANT_PRIOR,
            "excludes_current": True,
            "excludes_same_timestamp": True,
            "strictly_future": True,
            "amendment_version": AMENDMENT_VERSION,
        }]
        return self._block(
            signal.reshape(-1, 1),
            ["contam_M08_future_entity_rate"],
            ["M08"],
            params,
            metadata={
                "entity_ids": entity_ids,
                "m08_future_count": future_count,
                "m08_future_rate": future_rate,
            },
        )
