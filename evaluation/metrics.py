"""
metrics.py
----------
Standard IR metrics, kept dependency-free (no numpy required) since the
per-query lists here are always small (top-k <= a few hundred at most).
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of relevant_ids that appear anywhere in the top-k retrieved_ids."""
    if not relevant_ids:
        raise ValueError("relevant_ids must be non-empty")
    top_k = list(retrieved_ids)[:k]
    hits = len(set(top_k) & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    """1 / (rank of first relevant hit), 0.0 if none of relevant_ids are retrieved."""
    for idx, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / idx
    return 0.0


def mean_recall_at_k(
    pairs: Iterable[Tuple[Sequence[str], Set[str]]], k: int
) -> float:
    """pairs: iterable of (retrieved_ids, relevant_ids) across a whole eval set."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("pairs must be non-empty")
    return sum(recall_at_k(r, rel, k) for r, rel in pairs) / len(pairs)


def mean_reciprocal_rank(pairs: Iterable[Tuple[Sequence[str], Set[str]]]) -> float:
    pairs = list(pairs)
    if not pairs:
        raise ValueError("pairs must be non-empty")
    return sum(reciprocal_rank(r, rel) for r, rel in pairs) / len(pairs)


def heading_ids(retrieved_chunks, relevant_headings: List[str]) -> Tuple[List[str], Set[str]]:
    """
    Convenience: many docling-style pipelines are easier to eval by heading
    string than by opaque chunk_id, especially early on before you've pinned
    down stable IDs. Returns (retrieved_heading_list, relevant_heading_set)
    ready to feed into recall_at_k / reciprocal_rank.
    """
    return [c.headings for c in retrieved_chunks], set(relevant_headings)
