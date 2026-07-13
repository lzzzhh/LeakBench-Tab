"""benchmark_v2/policies/audit.py — Audit budget policies."""
from __future__ import annotations
import numpy as np
from enum import Enum
from benchmark_v2.core.models import DetectorOutput, LeakageLabel

class PolicyName(str, Enum):
    NO_REMOVAL = "no_removal"
    AUTO_QUARANTINE = "auto_quarantine"
    REMOVE_ALL_ORACLE = "remove_all_oracle"
    RANDOM_MATCHED = "random_matched"

def apply_policy(output, gt_dict, policy, audit_budget_pct=0.10, seed=42):
    rng = np.random.RandomState(seed)
    fids = list(output.feature_ids)
    ranking = list(output.audit_ranking)
    n = len(fids)
    k = int(np.ceil(audit_budget_pct * n))
    fid_by_rank = sorted(zip(ranking, fids))
    ranked_fids = [f for _, f in fid_by_rank]

    if policy == PolicyName.NO_REMOVAL:
        return fids, {"policy": "no_removal", "removed": 0, "kept": n}
    elif policy == PolicyName.AUTO_QUARANTINE:
        removed = set(ranked_fids[:k])
        kept = [f for f in fids if f not in removed]
        return kept, {"policy": "auto_quarantine", "removed": k, "kept": n-k}
    elif policy == PolicyName.REMOVE_ALL_ORACLE:
        removed = set()
        for fid in fids:
            orig = fid  # Use original IDs
            if gt_dict.get(orig) in (LeakageLabel.DIRECT_FORBIDDEN, LeakageLabel.PROXY, LeakageLabel.POST_OUTCOME):
                removed.add(fid)
        kept = [f for f in fids if f not in removed]
        return kept, {"policy": "remove_all_oracle", "removed": len(removed), "kept": len(kept)}
    elif policy == PolicyName.RANDOM_MATCHED:
        removed = set(rng.choice(fids, size=k, replace=False))
        kept = [f for f in fids if f not in removed]
        return kept, {"policy": "random_matched", "removed": k, "kept": n-k}
    raise ValueError(f"Unknown policy: {policy}")
