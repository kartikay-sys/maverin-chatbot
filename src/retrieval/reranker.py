import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class BGEReranker:
    """
    Cross-encoder reranker (BAAI/bge-reranker-v2-m3).
    Instead of just comparing vectors, a cross-encoder reads the Question + Chunk 
    simultaneously to determine true contextual relevance.
    """
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name, max_length=8192)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5):
        """
        Scores candidate chunks against the query and returns the most relevant top_k.
        Assumes candidates is a list of dictionaries containing a 'text' field.
        """
        if not candidates:
            return []
            
        # Format pairs for the cross-encoder: [[Query, Chunk1], [Query, Chunk2], ...]
        pairs = [[query, candidate['text']] for candidate in candidates]
        
        # Compute relevance scores
        scores = self.model.predict(pairs)
        
        # Handle case where only 1 candidate exists (compute_score returns a float instead of list)
        if isinstance(scores, float):
            scores = [scores]
            
        # Attach scores to the candidate payloads and sort descending
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            candidate['relevance_score'] = float(score)
            scored_candidates.append(candidate)
            
        scored_candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Return the absolute best chunks to pass to Gemini
        return scored_candidates[:top_k]
