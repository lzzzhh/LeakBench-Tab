"""T0-B Policy Views — oracle-isolated dataclasses for non-oracle selectors.

PolicyGroupView: the ONLY type non-oracle selectors receive.
SemanticEvaluationLabels: evaluation-only, NEVER passed to selectors.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyGroupView:
    """Oracle-isolated view of one semantic group.

    Contains NO leak_mask, contaminated_status, source_role, or mechanism category.
    Selector API receives ONLY: opaque_group_id, member_encoded_indices, group_size.
    """
    opaque_group_id: str
    member_encoded_indices: tuple[int, ...]  # sorted ascending
    group_size: int

    def __post_init__(self):
        if self.group_size != len(self.member_encoded_indices):
            raise ValueError(
                f"group_size {self.group_size} != len(member_encoded_indices) {len(self.member_encoded_indices)}"
            )
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")

    def encoded_columns(self) -> list[int]:
        return list(self.member_encoded_indices)


@dataclass(frozen=True)
class SemanticEvaluationLabels:
    """Evaluation-only contamination labels.

    MUST NOT be passed to P2-P6 policy selectors.
    Used only AFTER selection hash is produced, for computing
    leak_recall, semantic_group_recall, etc.
    """
    leak_mask: tuple[bool, ...]  # per-encoded-column boolean
    contaminated_group_ids: frozenset[str]
    legitimate_group_ids: frozenset[str]

    def is_contaminated_group(self, group_id: str) -> bool:
        return group_id in self.contaminated_group_ids

    def is_legitimate_group(self, group_id: str) -> bool:
        return group_id in self.legitimate_group_ids
