"""
test_retrieval.py
------------------
Recall@K and MRR for the Hybrid Search (bge-m3 dense+sparse, RRF fusion) +
bge-reranker-v2-m3 stage. Thresholds below are deliberately conservative
starting points -- tighten them once you have a real baseline from your
own pipeline; the point of a regression suite is to catch a *drop* from
today's number, not to enforce an arbitrary target on day one.
"""
from evaluation.metrics import mean_recall_at_k, mean_reciprocal_rank, recall_at_k, reciprocal_rank

RECALL_AT_5_THRESHOLD = 0.7
MRR_THRESHOLD = 0.5

# Acronym-heavy queries a telecom RAG system has to nail, since these are
# exact-match-shaped (not paraphrase-shaped) queries where sparse/BM25-style
# signal in the hybrid search matters as much as dense embeddings.
ACRONYM_QUERIES = [
    ("What does AMF stand for and what is its role?", "amf"),
    ("What is the function of the UPF in the 5G core?", "upf"),
    ("What is an SIB in the RRC protocol?", "sib"),
    ("What does the abbreviation PDU Session refer to?", "pdu session"),
]


def test_recall_at_5_per_query(pipeline, qa_pairs):
    failures = []
    for qa in qa_pairs:
        retrieved = pipeline.retrieve(qa.question, top_k=5)
        retrieved_ids = [c.chunk_id for c in retrieved]
        score = recall_at_k(retrieved_ids, set(qa.expected_source_chunk_ids), k=5)
        if score < 1.0:
            failures.append((qa.id, score, retrieved_ids))
    assert not failures, (
        f"{len(failures)} quer(ies) failed to retrieve their expected chunk "
        f"in top-5: {failures}"
    )


def test_mean_recall_at_5_across_dataset(pipeline, qa_pairs):
    pairs = [
        (
            [c.chunk_id for c in pipeline.retrieve(qa.question, top_k=5)],
            set(qa.expected_source_chunk_ids),
        )
        for qa in qa_pairs
    ]
    score = mean_recall_at_k(pairs, k=5)
    assert score >= RECALL_AT_5_THRESHOLD, (
        f"mean Recall@5 = {score:.2f}, below threshold {RECALL_AT_5_THRESHOLD}"
    )


def test_mean_reciprocal_rank_across_dataset(pipeline, qa_pairs):
    pairs = [
        (
            [c.chunk_id for c in pipeline.retrieve(qa.question, top_k=10)],
            set(qa.expected_source_chunk_ids),
        )
        for qa in qa_pairs
    ]
    score = mean_reciprocal_rank(pairs)
    assert score >= MRR_THRESHOLD, f"MRR = {score:.2f}, below threshold {MRR_THRESHOLD}"


def test_individual_reciprocal_rank_reported(pipeline, qa_pairs):
    """Not a pass/fail gate -- prints per-query MRR so a regression is
    traceable to a specific question instead of just an aggregate dip."""
    for qa in qa_pairs:
        retrieved = pipeline.retrieve(qa.question, top_k=10)
        rr = reciprocal_rank(
            [c.chunk_id for c in retrieved], set(qa.expected_source_chunk_ids)
        )
        print(f"[MRR] {qa.id}: {rr:.3f}")


def test_acronym_heavy_queries_surface_relevant_heading(pipeline):
    failures = []
    for query, expected_keyword in ACRONYM_QUERIES:
        retrieved = pipeline.retrieve(query, top_k=5)
        headings = " | ".join(c.headings.lower() for c in retrieved)
        text = " | ".join(c.text.lower() for c in retrieved)
        if expected_keyword not in headings and expected_keyword not in text:
            failures.append((query, expected_keyword, headings))
    assert not failures, f"Acronym queries with no relevant hit in top-5: {failures}"
