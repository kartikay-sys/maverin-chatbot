import pytest
from src.pipeline.query import RAGPipeline

# Initialize the pipeline once for all tests to save setup time
@pytest.fixture(scope="module")
def pipeline():
    return RAGPipeline()

# Expected strict fallback string
FALLBACK_STRING = "The provided 3GPP evidence is insufficient to answer this question."

LEAK_PHRASES = [
    "general knowledge",
    "general 3gpp knowledge",
    "supplementary information",
    "not sourced from the provided evidence",
    "based on standard 3gpp specifications"
]

def assert_no_leaks(answer: str):
    lower_ans = answer.lower()
    for phrase in LEAK_PHRASES:
        assert phrase not in lower_ans, f"Leak phrase '{phrase}' found in answer: {answer}"

def test_zero_retrieval_score_refusal(pipeline):
    """Regression: Topic not in text (zero/low retrieval). Must hard-refuse."""
    res = pipeline.ask("Summarize the key differences between a standalone and non-standalone 5G deployment.")
    assert res['answer'].strip() == FALLBACK_STRING
    assert_no_leaks(res['answer'])

def test_partial_evidence_refusal(pipeline):
    """Regression: Topic in text but limit is missing. Must hard-refuse."""
    res = pipeline.ask("What is the maximum number of PDU Sessions a UE can have with the same Single-NSSAI?")
    assert res['answer'].strip() == FALLBACK_STRING
    assert_no_leaks(res['answer'])

def test_topic_retrieved_missing_value_refusal(pipeline):
    """Guardrail: Topic in text but exact ms value is missing. Must hard-refuse."""
    res = pipeline.ask("What is the exact millisecond value of the PDU Session inactivity timer provided by the SMF to the UPF?")
    assert res['answer'].strip() == FALLBACK_STRING
    assert_no_leaks(res['answer'])

def test_wrong_domain_refusal(pipeline):
    """Guardrail: RAN/Layer 1 query in Core doc. Must hard-refuse."""
    res = pipeline.ask("What is the subcarrier spacing used for the 5G NR physical random access channel (PRACH)?")
    assert res['answer'].strip() == FALLBACK_STRING
    assert_no_leaks(res['answer'])

def test_valid_query_answers_normally_without_leaks(pipeline):
    """False-Refusal Check: Ensures normal queries still work and cite evidence."""
    res = pipeline.ask("What is the difference between the AMF and the SMF?")
    answer = res['answer'].strip()
    
    # Must NOT be a fallback refusal
    assert answer != FALLBACK_STRING, "Model falsely refused a query with valid evidence."
    assert_no_leaks(answer)
    
    # Should have some citation if it answered
    assert "[Evidence" in answer or "[evidence" in answer.lower(), "Valid answer lacked citations."
