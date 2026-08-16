import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class QdrantStore:
    """
    Manages the Qdrant vector database for Hybrid RAG.
    Stores dense vectors for semantic search and sparse vectors for BM25/keyword search.
    """
    def __init__(self, collection_name: str = "3gpp_docs", path: str = "data/qdrant_db"):
        self.collection_name = collection_name
        # Using local disk storage for reproducibility and easy testing
        self.client = QdrantClient(path=path)
        self._init_collection()

    def _init_collection(self):
        """Creates the collection with both dense and sparse configurations."""
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Creating new Qdrant Hybrid collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=1024, # BGE-M3 dimension
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )

    def insert_chunks(self, chunks: list[dict], dense_vecs: list, sparse_vecs: list, start_id: int = 0):
        """Upserts text chunks and their dense/sparse vectors into Qdrant."""
        points = []
        for i, (chunk, dense, sparse) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
            # Convert BGE sparse dictionary to Qdrant format
            indices = [int(k) for k in sparse.keys()]
            values = list(sparse.values())
            
            points.append(models.PointStruct(
                id=start_id + i, 
                payload={"text": chunk["text"], "metadata": chunk.get("metadata", {})},
                vector={
                    "dense": dense.tolist() if hasattr(dense, 'tolist') else dense,
                    "sparse": models.SparseVector(indices=indices, values=values)
                }
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} hybrid chunks to Qdrant.")

    def hybrid_search(self, dense_query: list, sparse_query: dict, limit: int = 20):
        """
        Executes a hybrid search using Qdrant's native Reciprocal Rank Fusion (RRF).
        This eliminates the need for a separate manual fusion script.
        """
        sparse_indices = [int(k) for k in sparse_query.keys()]
        sparse_values = list(sparse_query.values())
        
        # Define concurrent prefetch queries for dense (semantic) and sparse (keyword)
        prefetch = [
            models.Prefetch(
                query=dense_query.tolist() if hasattr(dense_query, 'tolist') else dense_query,
                using="dense",
                limit=limit
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=limit
            )
        ]
        
        # Qdrant natively fuses the results using RRF
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True
        )
        
        # Return the extracted payloads (text + metadata) for the Reranker
        return [res.payload for res in results.points]
