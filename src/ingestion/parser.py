import json
import logging
from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter

# Configure basic logging for the ingestion pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DoclingIngestor:
    """
    Handles the ingestion of raw DOCX/PDF files using Docling.
    Extracts text, preserves structural layout, and extracts complex tables.
    """
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/parsed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # DocumentConverter natively parses complex structures, layouts, and tables out-of-the-box.
        self.converter = DocumentConverter()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_document_paths(self) -> List[Path]:
        """Retrieves all valid DOCX and PDF documents from the input directory."""
        paths = []
        for ext in ['*.docx', '*.pdf']:
            for file_path in self.input_dir.glob(ext):
                # Ignore temporary Word files
                if not file_path.name.startswith('~$'):
                    paths.append(file_path)
        return paths

    def ingest_document(self, file_path: Path) -> bool:
        """
        Parses a single document and exports both Markdown (for readability/basic chunking) 
        and JSON (to preserve rich layout, tables, and bounding boxes).
        """
        logger.info(f"Ingesting document: {file_path.name}")
        try:
            result = self.converter.convert(str(file_path))
            
            # Export Markdown format
            md_path = self.output_dir / f"{file_path.stem}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(result.document.export_to_markdown())
            
            # Export native JSON dict to preserve deep structure for downstream processing
            json_path = self.output_dir / f"{file_path.stem}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result.document.export_to_dict(), f, indent=2)
                
            logger.info(f"Successfully processed and saved outputs for {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest {file_path.name}: {str(e)}")
            return False

    def run_pipeline(self):
        """Executes the ingestion pipeline on all valid documents."""
        docs = self.get_document_paths()
        if not docs:
            logger.warning(f"No valid documents found in {self.input_dir}")
            return
            
        logger.info(f"Found {len(docs)} documents. Starting ingestion...")
        
        success_count = 0
        for doc in docs:
            if self.ingest_document(doc):
                success_count += 1
                
        logger.info(f"Ingestion complete. Successfully parsed {success_count}/{len(docs)} documents.")

if __name__ == "__main__":
    # Execution entry point for the ingestion stage
    ingestor = DoclingIngestor()
    ingestor.run_pipeline()
