import json
import logging
from pathlib import Path
from typing import List

# Use docling's native structural chunker and core document model
from docling.chunking import HierarchicalChunker
from docling_core.types.doc import DoclingDocument

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StructureAwareChunker:
    """
    Reads parsed Docling JSON documents and applies structure-aware chunking.
    This ensures that 3GPP sections, subsections, and tables are kept intact 
    rather than arbitrarily split by word count.
    """
    def __init__(self, input_dir: str = "data/parsed", output_dir: str = "data/chunks"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # HierarchicalChunker natively respects document structure (headings, tables, lists)
        self.chunker = HierarchicalChunker()

    def get_json_paths(self) -> List[Path]:
        """Retrieves all parsed JSON documents."""
        return list(self.input_dir.glob("*.json"))

    def process_document(self, file_path: Path) -> bool:
        """Loads a parsed JSON document, chunks it structurally, and saves the chunks."""
        logger.info(f"Chunking document: {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_dict = json.load(f)
            
            # Reconstruct the DoclingDocument from the saved JSON dictionary
            doc = DoclingDocument.model_validate(doc_dict)
            
            # Perform structural chunking using Docling's native logic
            chunks = list(self.chunker.chunk(doc))
            
            # Serialize the chunks for the downstream vector database (Qdrant)
            serialized_chunks = []
            for c in chunks:
                serialized_chunks.append({
                    "text": c.text,
                    # We preserve the metadata (e.g. heading paths) to help with retrieval
                    "metadata": c.meta.model_dump()
                })
                
            output_path = self.output_dir / f"{file_path.stem}_chunks.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serialized_chunks, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Created {len(chunks)} structural chunks for {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to chunk {file_path.name}: {str(e)}")
            return False

    def run_pipeline(self):
        """Executes the chunking pipeline on all available parsed documents."""
        docs = self.get_json_paths()
        if not docs:
            logger.warning(f"No parsed JSON documents found in {self.input_dir}")
            return
            
        logger.info(f"Starting structure-aware chunking for {len(docs)} documents...")
        for doc in docs:
            self.process_document(doc)
        logger.info("Chunking pipeline complete.")

if __name__ == "__main__":
    chunker = StructureAwareChunker()
    chunker.run_pipeline()
