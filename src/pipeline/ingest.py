import os
import glob
import json
import logging
from pathlib import Path

# Ensure src modules can be imported
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.retrieval.embeddings import BGEM3Embedder
from src.retrieval.vector_store import QdrantStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def populate_database(chunks_dir: str = "data/chunks"):
    """
    Reads the serialized JSON chunks from the chunking stage, 
    embeds them using BGE-M3, and upserts them into Qdrant.
    """
    logger.info("Initializing Embedder and Vector Store...")
    embedder = BGEM3Embedder()
    store = QdrantStore()
    
    chunk_files = glob.glob(os.path.join(chunks_dir, "*_chunks.json"))
    if not chunk_files:
        logger.warning(f"No chunk files found in {chunks_dir}.")
        return

    current_id = 0
    for chunk_file in chunk_files:
        logger.info(f"Processing chunks from {os.path.basename(chunk_file)}")
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        if not chunks:
            continue
            
        texts = [c["text"] for c in chunks]
        
        # We encode in batches to avoid OOM errors on the GPU/CPU
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]
            
            logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch_texts)} chunks)")
            dense_vecs, sparse_vecs = embedder.encode(batch_texts)
            
            store.insert_chunks(
                chunks=batch_chunks,
                dense_vecs=dense_vecs,
                sparse_vecs=sparse_vecs,
                start_id=current_id
            )
            current_id += len(batch_texts)
            
    logger.info(f"Database population complete! Upserted {current_id} total chunks.")

if __name__ == "__main__":
    populate_database()
