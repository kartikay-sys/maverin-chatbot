@echo off
echo ==================================================
echo Starting 3GPP RAG Assistant
echo ==================================================

echo Loading local models and starting FastAPI backend...
uvicorn src.api.main:app --reload

pause
