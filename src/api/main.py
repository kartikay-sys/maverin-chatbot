import os
import sys
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipeline.query import RAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="3GPP RAG API")

pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        logger.info("Starting up FastAPI and loading RAG pipeline...")
        pipeline = RAGPipeline()
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Handles chat requests by querying the RAG pipeline."""
    if not pipeline:
        return {"error": "Pipeline not initialized."}
        
    try:
        result = pipeline.ask(req.message)
        return {
            "answer": result["answer"],
            "evidence": result["evidence"]
        }
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        return {"error": str(e)}

# Serve static files from the frontend directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")
