# Maverin: A chatbot on Telecom 3GPP standards

Maverin is a specialized Retrieval-Augmented Generation (RAG) system engineered specifically to navigate, search, and answer questions based on highly technical 3GPP telecom specifications. 

By combining hierarchical document parsing, hybrid vector search, and a strict conversational guardrail system, Maverin delivers accurate, grounded answers to complex network architecture and protocol questions without the risk of AI hallucination.
## 📚 Primary Knowledge Source

- **3GPP TS 23.501 V20.2.0 (2026-06)**
  - *System architecture for the 5G System (5GS); Stage 2*
  - 3GPP Release 20
  - Primary source for 5G System architecture, network functions, registration, connection management, session management, QoS, network slicing, etc.

## 📸 Screenshots

<img width="1920" height="912" alt="image" src="https://github.com/user-attachments/assets/47a69076-67fa-45ce-b330-75418c7cf50f" />
<img width="1920" height="912" alt="image" src="https://github.com/user-attachments/assets/0007ed56-4ccf-4662-a1be-ab74390f0090" />


## 🌟 Key Features

- **Structural Document Chunking**: Powered by [Docling](https://github.com/DS4SD/docling), the ingestion pipeline understands document hierarchies, preserving 3GPP sections, headers, and structural integrity rather than blindly splitting by word count.
- **Hybrid Search Engine**: Uses a local **Qdrant** vector database with **BAAI/bge-m3** embeddings, combining dense semantic search with sparse lexical search (BM25) via Reciprocal Rank Fusion (RRF) for unparalleled retrieval accuracy.
- **Advanced Reranking**: Re-scores the top candidates using a cross-encoder (**BGE Reranker**) to ensure the absolute most relevant chunks are fed to the LLM.
- **Zero-Hallucination Guardrails**: Powered by **NVIDIA's Nemotron-3** model, the LLM is strictly constrained via specialized prompt engineering to answer *only* based on the retrieved 3GPP evidence. If the context is missing, it falls back safely.
- **Clean Chat UI**: A lightweight, fast, and responsive vanilla JS/HTML frontend served by FastAPI.

## 🛠️ Architecture

- **Backend**: FastAPI (Python)
- **Vector Store**: Qdrant (Local persistent DB)
- **Embeddings**: BAAI/bge-m3
- **Reranker**: BAAI/bge-reranker-large
- **LLM Engine**: NVIDIA API (Nemotron-3-ultra)
- **Frontend**: Vanilla HTML/CSS/JS

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- An NVIDIA API Key

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/maverin.git
cd maverin
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the root of the project and add your NVIDIA API key:

```env
NVIDIA_API_KEY="your_nvidia_api_key_here"
```

### 3. Data Ingestion

Before starting the app, you need to parse and embed the 3GPP specification documents (e.g., `.docx` files) into the local Qdrant database. Place your 3GPP documents in the appropriate data folder, then run the ingestion pipeline:

```bash
python -m src.pipeline.ingest
```
*Note: This process uses Docling to hierarchically chunk the documents and BGE-M3 to embed them. It may take a few minutes depending on document size.*

### 4. Run the Application

Start the FastAPI backend and frontend server:

```bash
# On Windows
.\run.bat
```

Open your browser and navigate to **`http://localhost:8000`** to chat with Maverin.



## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
