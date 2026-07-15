# SP8-B Sentinel Governance Report

**Models: LR** (pilot; catboost/tabm pending) | **672 cells** | **8s**

## Key Findings

### Field-Score Budget (P3) — Primary Non-Oracle Policy
| Budget | Leak Recall | Legit Retention | Residual Harm |
|--------|------------|----------------|--------------|
| 5% | 0.45 | 0.99 | −0.12 |
| 10% | 0.58 | 0.98 | −0.17 |
| 20% | 0.78 | 0.93 | −0.26 |

Oracle upper bound: recall 1.0, retention 1.0, residual −0.33.

### Per-Mechanism (20% budget)
- M06 (redundant): recall 0.83, retention 0.98 (easy to detect)
- M09 (source): recall 0.56, retention 0.96 (moderate)
- M10 (mixed): recall 1.0, retention 0.82 (high recall at legit cost)
- M11 (graph): recall 0.73, retention 0.96 (good)

### Strategy Comparison
- Field budget (P3): recall 0.60, retention 0.97, residual −0.18
- Group budget (P4): recall 0.60, retention 1.0, residual −0.08
- Lifecycle (P5): N/A for governance mechs (applies only to temporal)
- Oracle (P1): recall 1.0, residual −0.33 (upper bound)

## Conclusion
Blind field-score governance at 20% budget removes 78% of leaked features
while retaining 93% of legitimate ones, achieving 78% of oracle harm reduction.
Group-budget preserves all legitimate features at the cost of weaker recall.
Simple blind MI is the most practical governance signal.
