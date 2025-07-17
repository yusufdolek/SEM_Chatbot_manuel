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
- **High token usage:** Parent-child retrieval sends entire parent documents (~21K tokens per query)

## Debug Tools

### **Token Analysis:**
- **Script:** `test_token.py` - Analyzes total token usage of company documents
- **Usage:** `python3 test_token.py`
- **Output:** Token counts, cost estimation, optimization recommendations

### **Chunk Analysis:**
- **Script:** `debug_chunks.py` - Exports all chunks and analyzes specific queries
- **Usage:** `python3 debug_chunks.py`
- **Output:** `chunks.txt` file with all chunks (⚠️ DELETE after use - contains sensitive data)
- **Purpose:** Debug chunk quality and token usage for specific queries

### **Performance Issues:**
- **SmartFeed query problem:** Uses ~21K tokens due to large parent documents
- **Root cause:** Parent-child retrieval returns entire parent documents instead of relevant chunks
- **Solution:** Implement child-only retrieval with higher similarity thresholds

## Token Usage Optimizations (IMPLEMENTED)

### **Before Optimization:**
- **SmartFeed query:** ~21,019 tokens (94,565 characters)
- **Method:** Parent-child retrieval returning entire parent documents
- **Similarity threshold:** 0.25 (very low)
- **Results:** 2 large parent documents

### **After Optimization:**
- **SmartFeed query:** ~387 tokens (1,633 characters)
- **Method:** Child-only retrieval with context length limits
- **Similarity threshold:** 0.4 (higher quality)
- **Results:** 5 relevant chunks
- **Improvement:** **98.2% token reduction** (54x less tokens)

### **Optimization Changes Made:**

**1. Vector Store Modifications (`vector_store.py`):**
- Changed from returning parent documents to returning only relevant chunks
- Increased similarity threshold from 0.25 to 0.4 for better quality
- Added context length limit (15,000 characters max)
- Reduced top_k from 10 to 5 results

**2. Chatbot Logic Updates (`chatbot.py`):**
- Updated to handle chunk-based retrieval instead of parent documents
- Added token usage logging for monitoring
- Maintained enhanced search logic for brand queries
- Preserved fallback mechanisms with optimized thresholds

**3. Performance Improvements:**
- **Token usage:** 21,019 → 387 tokens (98.2% reduction)
- **Context size:** 94,565 → 1,633 characters (98.3% reduction)
- **Cost reduction:** ~$0.04 → ~$0.0008 per query (95% cost savings)
- **Response quality:** Maintained with more focused, relevant chunks

### **Key Metrics:**
- **Average chunk size:** 360 characters
- **Total chunks in system:** 397
- **Estimated total tokens:** 31,698
- **Context limit:** 15,000 characters
- **Similarity threshold:** 0.4 (fallback: 0.25)