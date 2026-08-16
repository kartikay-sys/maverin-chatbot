import logging
from src.retrieval.embeddings import BGEM3Embedder
from src.retrieval.vector_store import QdrantStore
from src.retrieval.reranker import BGEReranker
from src.generation.llm import NvidiaGenerator

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Ties together the Retrieval and Generation components into a single end-to-end flow.
    """
    def __init__(self):
        logger.info("Initializing RAG Pipeline...")
        self.embedder = BGEM3Embedder()
        self.vector_store = QdrantStore()
        self.reranker = BGEReranker()
        self.generator = NvidiaGenerator()
        logger.info("RAG Pipeline ready.")

    def ask(self, question: str) -> dict:
        """
        Executes the full RAG pipeline for a given user question.
        Returns the generated answer and the top evidence used.
        """
        logger.info(f"Processing question: {question}")
        
        # 1. Embed the question (Dense + Sparse)
        # Note: BGE-M3 encode expects a list of strings
        dense_vecs, sparse_vecs = self.embedder.encode([question])
        dense_query = dense_vecs[0]
        sparse_query = sparse_vecs[0]
        
        # 2. Hybrid Retrieval (Qdrant + RRF)
        # Retrieves a broader set of candidates (~20)
        retrieved_candidates = self.vector_store.hybrid_search(
            dense_query=dense_query, 
            sparse_query=sparse_query, 
            limit=20
        )
        
        # 3. Cross-Encoder Reranking
        # Narrows down to the absolute best top 5 chunks
        top_evidence = self.reranker.rerank(
            query=question, 
            candidates=retrieved_candidates, 
            top_k=5
        )
        
        # 4. Gemini Generation
        answer = self.generator.generate_answer(
            question=question, 
            retrieved_chunks=top_evidence
        )
        
        return {
            "answer": answer,
            "evidence": top_evidence
        }
