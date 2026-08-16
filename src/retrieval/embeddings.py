import logging
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)

class BGEM3Embedder:
    """
    Wraps BAAI/bge-m3 for dense and sparse embeddings.
    BGE-M3 is ideal for 3GPP because it supports 8,192 token contexts 
    and natively produces both semantic (dense) and lexical (sparse/BM25-like) vectors.
    """
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        logger.info(f"Loading embedding model: {model_name}")
        # use_fp16 reduces memory usage without significant quality loss
        self.model = BGEM3FlagModel(model_name, use_fp16=True)

    def encode(self, texts: list[str]):
        """
        Generates both dense vectors (for semantic search) 
        and sparse vectors (for exact keyword/BM25 matching).
        """
        # We explicitly request both dense and sparse representations
        output = self.model.encode(
            texts, 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )
        return output['dense_vecs'], output['lexical_weights']
