# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
python app.py
```
Server runs on http://localhost:5001 with auto-reload enabled.

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Required Environment Variables
Create `.env` file in project root:
```env
GEMINI_API_KEY="AIzaSy..."
TOKENIZERS_PARALLELISM=false
```

### Vector Database Management
- FAISS index files are auto-generated on first run from PDFs in `company_docs/`
- To rebuild index: delete `faiss_index.bin`, `faiss_child_metadata.pkl`, `faiss_parent_docstore.pkl`
- Index building happens at app startup via `setup_vector_store()`

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) chatbot** for SEM company's corporate knowledge base, built with FastAPI and async architecture.

### Core RAG Flow
1. **Document Loading** (`document_loader.py`): PDFs → Logical sections → Character chunks
2. **Embedding** (`embedding.py`): Text → Vectors via SentenceTransformers
3. **Vector Search** (`vector_store.py`): Query → Relevant document chunks via FAISS
4. **Generation** (`llm.py`): Context + Query → Response via Gemini
5. **Media Enhancement** (`media_mapping.py`): Response → Enhanced with images/videos

### Key Components

**Main Application (`app.py`)**
- FastAPI server with `/chat` endpoint and static file serving
- Async request handling with proper error management

**RAG Pipeline (`rag_chatbot/`)**
- `chatbot.py`: Main orchestrator with enhanced search logic
- `document_loader.py`: **Critical hybrid chunking strategy**
- `vector_store.py`: FAISS operations with parent-child document mapping
- `llm.py`: Gemini API integration with system prompts
- `embedding.py`: SentenceTransformers with brand name normalization
- `media_mapping.py`: Static media URL mapping for contextual enhancement

**Frontend (`templates/index.html`)**
- Chat widget with lightbox image viewing
- Markdown rendering and media embedding support

### Critical Implementation Details

**Hybrid Chunking Strategy (Most Important)**
- **Problem**: Standard chunking loses context or creates oversized chunks
- **Solution**: Two-stage approach in `document_loader.py`:
  1. Split by logical sections (SECTION 1, Case Studies) using regex
  2. Split sections into 400-character chunks with 100-character overlap
  3. Maintain parent-child relationships for context preservation

**Enhanced Search Logic**
- Brand-specific query expansion for companies (Beymen, Migros, etc.)
- Multi-threshold fallback mechanism (0.25 → 0.15)
- Document prioritization for detailed case studies
- Async thread pool execution for CPU-bound operations

**Vector Database Structure**
- `faiss_index.bin`: Vector embeddings (child documents)
- `faiss_child_metadata.pkl`: Child document metadata
- `faiss_parent_docstore.pkl`: Full parent document content
- Uses `IndexFlatIP` for cosine similarity with score thresholds

**Media Integration**
- Static mapping in `media_mapping.py` for brands/services
- Automatic YouTube URL → embed URL conversion
- Contextual media display based on query keywords

### Technical Stack
- **Backend**: FastAPI (async), Python 3.13+
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector DB**: FAISS with cosine similarity
- **Document Processing**: PyMuPDF + LangChain
- **Frontend**: Vanilla JavaScript with Markdown rendering

### Domain-Specific Knowledge

**Turkish Corporate Context**
- Handles Turkish company documents with section numbering
- Brand name normalization for better semantic matching
- Bilingual support (Turkish/English) with language detection

**SEM-Specific Features**
- Media mapping for SEM products (SmartFeed, Data Bridge)
- Case study prioritization for brand queries
- Google Marketing Platform service integration

## File Structure Notes

- `company_docs/`: PDF documents are processed from here
- `static/`: Frontend assets including chat widget CSS
- `rag_chatbot/`: All RAG pipeline components
- Vector index files are generated at project root

## Known Issues

- Image URLs from `webtest.semtr.com` may return 421 errors
- Vector database rebuilds on missing index files (not incremental)
- No chat history persistence between sessions