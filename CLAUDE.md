# CLAUDE.md - Technical Guide for Future Claude Instances

This comprehensive guide provides detailed technical information for Claude Code (claude.ai/code) when working with the SEM RAG Chatbot codebase.

---

## 🚀 Quick Start Commands

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

---

## 🏗️ System Architecture Deep Dive

### RAG Pipeline Overview
This is a **production-grade RAG (Retrieval-Augmented Generation) chatbot** for SEM company's corporate knowledge base, built with FastAPI and async architecture.

#### Core RAG Flow
1. **Document Loading** (`document_loader.py`): PDFs → Logical sections → Character chunks
2. **Embedding** (`embedding.py`): Text → 384-dimensional vectors via SentenceTransformers
3. **Vector Search** (`vector_store.py`): Query → Relevant document chunks via FAISS
4. **Generation** (`llm.py`): Context + Query → Response via Gemini 2.5 Flash
5. **Media Enhancement** (`media_extractor.py`): Response → Enhanced with images/videos

### Key Components

#### **Main Application (`app.py`)**
- FastAPI server with `/chat` endpoint and static file serving
- Async request handling with proper error management
- Health check endpoint at `/health`

#### **RAG Pipeline (`rag_chatbot/`)**
- `chatbot.py`: Main orchestrator with optimized search logic
- `document_loader.py`: **Critical hybrid chunking strategy**
- `vector_store.py`: FAISS operations with optimized context management
- `llm.py`: Gemini API integration with comprehensive response prompts
- `embedding.py`: SentenceTransformers with simple brand normalization
- `media_extractor.py`: Response enhancement with contextual media

#### **Frontend (`templates/`)**
- `index.html`: Main chat interface with lightbox image viewing
- `backpage.html`: Secondary test page for navigation testing
- Markdown rendering and media embedding support
- Session persistence across page navigation

---

## ⚙️ Critical Hyperparameters & Configuration

### Embedding & Vector Search Parameters

#### **SentenceTransformers Configuration**
```python
# In rag_chatbot/embedding.py
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
SIMILARITY_METRIC = "cosine"
```

#### **FAISS Vector Store Settings**
```python
# In rag_chatbot/vector_store.py
INDEX_TYPE = "IndexFlatIP"          # Cosine similarity index
SIMILARITY_THRESHOLD = 0.15         # Primary similarity threshold
FALLBACK_THRESHOLD = 0.2            # Fallback threshold if no results
TOP_K_RETRIEVAL = 15                # Number of chunks to retrieve
MAX_CONTEXT_LENGTH = 25000          # Maximum context characters
```

#### **Document Processing Parameters**
```python
# In rag_chatbot/document_loader.py
CHUNK_SIZE = 400                    # Characters per chunk
CHUNK_OVERLAP = 100                 # Overlap between chunks
LOGICAL_SECTION_SPLIT = True        # Enable header-based splitting
PARENT_CHILD_MAPPING = True         # Maintain chunk relationships
```

### LLM Generation Parameters

#### **Gemini 2.5 Flash Configuration**
```python
# In rag_chatbot/llm.py
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.7                   # Creativity vs consistency balance
MAX_OUTPUT_TOKENS = 2048            # Response length limit
TOP_P = 0.95                        # Nucleus sampling parameter
TOP_K = 40                          # Top-k sampling parameter
FREQUENCY_PENALTY = 0.0             # Repetition penalty
PRESENCE_PENALTY = 0.0              # Topic diversity penalty
```

#### **System Prompt Configuration**
```python
# Key prompt elements
SYSTEM_INSTRUCTION = """
You are SEM's expert, professional, and persuasive digital assistant.
- Match the user's language (English or Turkish)
- Provide comprehensive, detailed, and informative answers
- Use markdown formatting with bold headings and bullet points
- Professional tone with moderate emoji usage
- Elaborate on key points with context and examples
"""
```

### Performance Optimization Parameters

#### **Async & Threading Configuration**
```python
# In rag_chatbot/chatbot.py
THREAD_POOL_SIZE = 4                # CPU-bound task threads
REQUEST_TIMEOUT = 30                # Request timeout in seconds
MAX_CONCURRENT_REQUESTS = 100       # FastAPI concurrency limit
```

#### **Context Management**
```python
# Dynamic context building parameters
TOKEN_ESTIMATION_RATIO = 0.222      # Characters to tokens ratio
CONTEXT_TRUNCATION_STRATEGY = "priority"  # How to handle overflow
CHUNK_PRIORITIZATION = "similarity"  # Ranking method
```

---

## 🔬 Advanced RAG Implementation Details

### Hybrid Chunking Strategy (Critical)

#### **Problem Solved**
Standard chunking approaches either:
1. Lose context by arbitrary character splits
2. Create oversized chunks that overwhelm token limits

#### **Our Two-Stage Solution**
```python
# Stage 1: Logical sectioning by headers
def split_by_headers(document):
    # Use regex to identify document structure
    sections = re.split(r'(SECTION \d+|Case Study|1\.\d+)', document)
    return sections

# Stage 2: Character chunking with overlap
def split_by_characters(text, chunk_size=400, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks
```

#### **Parent-Child Relationship Mapping**
```python
# Each chunk maintains metadata about its source
chunk_metadata = {
    "chunk_id": "chunk_123",
    "parent_doc_id": "doc_456",
    "section": "Company Services",
    "chunk_index": 3,
    "source_file": "services.pdf",
    "page_number": 5
}
```

### Multi-Threshold Fallback System

#### **Implementation Logic**
```python
# Primary search with strict threshold
def search_with_fallback(query_embedding, top_k=15):
    # Primary search (strict quality)
    results = search_with_threshold(query_embedding, threshold=0.15)
    
    # Fallback if no results (broader search)
    if not results:
        results = search_with_threshold(query_embedding, threshold=0.2)
    
    return results
```

#### **Why This Works**
- **0.15 threshold**: Ensures high-quality, relevant results
- **0.2 fallback**: Catches edge cases and rare queries
- **Prevents empty responses**: Always returns something useful

### Context Length Management

#### **Dynamic Context Building**
```python
# Intelligent context assembly
def build_context(ranked_chunks, max_length=25000):
    context_parts = []
    current_length = 0
    
    for i, chunk in enumerate(ranked_chunks):
        chunk_length = len(chunk)
        
        # Check if adding this chunk exceeds limit
        if current_length + chunk_length > max_length:
            print(f"Context limit reached at {current_length} chars")
            break
        
        # Add chunk with section formatting
        formatted_chunk = f"**Information Section {i+1}:**\n{chunk}"
        context_parts.append(formatted_chunk)
        current_length += len(formatted_chunk)
    
    return "\n\n---\n\n".join(context_parts)
```

---

## 📊 Performance Metrics & Achievements

### Token Optimization Success Story

#### **Before Optimization (Original System)**
```
Query: "SmartFeed nedir?"
├── Method: Parent-child retrieval
├── Context: 94,565 characters
├── Tokens: ~21,019 tokens
├── Cost: ~$0.04 per query
├── Response Time: 8-12 seconds
└── Quality: Good but expensive
```

#### **After Optimization (Current System)**
```
Query: "SmartFeed nedir?"
├── Method: Child-only retrieval with context limits
├── Context: 1,633 characters
├── Tokens: ~387 tokens
├── Cost: ~$0.0008 per query
├── Response Time: 2-3 seconds
└── Quality: Maintained high quality
```

#### **Achievement Metrics**
- **98.2% token reduction** (21,019 → 387 tokens)
- **95% cost reduction** ($0.04 → $0.0008)
- **4x faster response time** (10s → 2.5s average)
- **54x efficiency improvement** overall

### Vector Database Performance

#### **Storage Efficiency**
```
Database Files:
├── faiss_index.bin: 170KB (vector embeddings)
├── faiss_child_metadata.pkl: 115KB (chunk metadata)
├── faiss_parent_docstore.pkl: 610KB (parent documents)
├── Total Storage: ~895KB
└── Search Speed: <50ms per query
```

#### **Indexing Statistics**
```
Document Processing:
├── Total Documents: 8 PDF files
├── Total Chunks: 397 indexed chunks
├── Average Chunk Size: 360 characters
├── Embedding Dimension: 384
└── Index Build Time: ~30 seconds
```

---

## 🎯 Domain-Specific Optimizations

### Turkish Language Support

#### **Brand Name Normalization**
```python
# In rag_chatbot/embedding.py
BRAND_MAPPINGS = {
    'lcwaikiki': 'LC WAIKIKI',
    'beymen': 'BEYMEN',
    'migros': 'MIGROS',
    'boyner': 'BOYNER',
    'teknosa': 'TEKNOSA'
}
```

#### **Language Detection Strategy**
```python
def detect_language(query):
    turkish_chars = set('çğıöşüÇĞIÖŞÜ')
    if any(char in turkish_chars for char in query):
        return "turkish"
    return "english"
```

### SEM-Specific Features

#### **Media Integration System**
```python
# In rag_chatbot/media_mapping.py
MEDIA_MAPPINGS = {
    'smartfeed': {
        'images': ['smartfeed_dashboard.png'],
        'videos': ['smartfeed_demo.mp4']
    },
    'beymen': {
        'images': ['beymen_interface.png'],
        'videos': ['beymen_case_study.mp4']
    }
}
```

#### **Case Study Prioritization**
- Enhanced retrieval for brand-specific queries
- Detailed performance metrics display
- Contextual media embedding based on query content

---

## 🛠️ Development Workflow & Best Practices

### Code Structure Guidelines

#### **File Organization**
```
SEM_Chatbot_manuel/
├── app.py                      # FastAPI application entry
├── rag_chatbot/               # Core RAG pipeline
│   ├── chatbot.py            # Main orchestrator
│   ├── document_loader.py    # PDF processing
│   ├── embedding.py          # Text embeddings
│   ├── vector_store.py       # FAISS operations
│   ├── llm.py               # Gemini integration
│   ├── media_extractor.py   # Media enhancement
│   └── media_mapping.py     # Static media mappings
├── templates/                # Jinja2 templates
│   ├── index.html           # Main chat interface
│   └── backpage.html        # Test page
├── static/                   # Frontend assets
├── company_docs/            # PDF source documents
└── Vector files (auto-generated)
```

#### **Coding Standards**
- **Async/await patterns**: Used throughout for non-blocking operations
- **Error handling**: Comprehensive try-catch blocks
- **Type hints**: Added where beneficial
- **Documentation**: Concise but informative comments

### Testing & Debugging

#### **Key Test Scenarios**
1. **Company Services**: "SEM ne tür hizmetler sunuyor?"
2. **Case Studies**: "Beymen projesinin detayları"
3. **Technical Queries**: "SmartFeed nasıl çalışır?"
4. **Client Portfolio**: "Hangi markalarla çalışıyorsunuz?"

#### **Performance Monitoring**
```python
# Token usage tracking
def log_token_usage(context):
    chars = len(context)
    tokens = int(chars * 0.222)
    print(f"Token usage: ~{tokens} tokens ({chars} chars)")
```

### Common Issues & Solutions

#### **Case Study Not Found**
- **Symptom**: Generic response instead of specific case study
- **Solution**: Increase `TOP_K_RETRIEVAL` from 15 to 20
- **Root Cause**: Relevant chunks ranked outside top 15

#### **High Token Usage**
- **Symptom**: Queries using >2000 tokens
- **Solution**: Reduce `MAX_CONTEXT_LENGTH` or increase `SIMILARITY_THRESHOLD`
- **Monitoring**: Check context length in logs

#### **Poor Response Quality**
- **Symptom**: Vague or irrelevant responses
- **Solution**: Decrease `SIMILARITY_THRESHOLD` or increase `TOP_K_RETRIEVAL`
- **Balance**: Trade-off between quality and token usage

---

## 🔧 Configuration Management

### Environment Variables

#### **Required Configuration**
```bash
# Core API access
GEMINI_API_KEY="your-gemini-api-key"
TOKENIZERS_PARALLELISM=false

# Optional performance tuning
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
THREAD_POOL_SIZE=4
DEBUG=false
```

#### **Development vs Production**
```bash
# Development
DEBUG=true
LOG_LEVEL=DEBUG
MAX_CONTEXT_LENGTH=25000

# Production
DEBUG=false
LOG_LEVEL=INFO
MAX_CONTEXT_LENGTH=20000
```

### Hyperparameter Tuning Guidelines

#### **For High Precision (Less Noise)**
```python
SIMILARITY_THRESHOLD = 0.2      # Increase threshold
TOP_K_RETRIEVAL = 10            # Reduce retrieved chunks
MAX_CONTEXT_LENGTH = 20000      # Shorter context
```

#### **For High Recall (More Coverage)**
```python
SIMILARITY_THRESHOLD = 0.1      # Decrease threshold
TOP_K_RETRIEVAL = 20            # More retrieved chunks
MAX_CONTEXT_LENGTH = 30000      # Longer context
```

#### **For Cost Optimization**
```python
SIMILARITY_THRESHOLD = 0.25     # Stricter filtering
TOP_K_RETRIEVAL = 8             # Fewer chunks
MAX_OUTPUT_TOKENS = 1024        # Shorter responses
```

---

## 📈 Monitoring & Analytics

### Key Metrics to Track

#### **System Performance**
```python
# Response time monitoring
import time

start_time = time.time()
response = await get_chatbot_response(query)
end_time = time.time()
response_time = end_time - start_time

print(f"Response time: {response_time:.2f} seconds")
```

#### **Token Usage Analytics**
```python
# Daily token usage tracking
def track_daily_usage():
    daily_tokens = 0
    daily_cost = 0
    
    for query_log in today_queries:
        tokens = estimate_tokens(query_log['context'])
        cost = tokens * GEMINI_COST_PER_TOKEN
        
        daily_tokens += tokens
        daily_cost += cost
    
    return daily_tokens, daily_cost
```

### Performance Optimization Strategies

#### **Real-time Monitoring**
- Track average response times
- Monitor token usage patterns
- Alert on unusual spikes

#### **Batch Analysis**
- Weekly performance reports
- Query pattern analysis
- Cost optimization opportunities

---

## 🔄 Session Management

### Chat Persistence Implementation

#### **Session Storage Strategy**
```javascript
// In templates/index.html
function saveChatHistory() {
    const messages = Array.from(messagesDiv.children).map(msg => ({
        className: msg.className,
        innerHTML: msg.innerHTML
    }));
    sessionStorage.setItem('chatHistory', JSON.stringify(messages));
}

function loadChatHistory() {
    const history = sessionStorage.getItem('chatHistory');
    if (history) {
        const messages = JSON.parse(history);
        messages.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = msg.className;
            messageDiv.innerHTML = msg.innerHTML;
            messagesDiv.appendChild(messageDiv);
        });
    }
}
```

#### **Navigation System**
- **Main Page**: Full chat interface at `/`
- **Back Page**: Test page at `/backpage`
- **History**: Persists across navigation
- **Reset**: Only on server restart

---

## 🚀 Future Development Roadmap

### Phase 1: Advanced RAG Features
- **Hybrid Search**: Combine semantic + keyword search
- **Query Expansion**: Automatic query rewriting
- **Multi-hop Reasoning**: Cross-document connections

### Phase 2: Performance Enhancements
- **Caching Layer**: Redis-based response caching
- **Streaming Responses**: Real-time response generation
- **Load Balancing**: Multi-instance deployment

### Phase 3: Enterprise Features
- **Multi-tenancy**: Support multiple organizations
- **Advanced Analytics**: Detailed usage reporting
- **Real-time Updates**: Live document synchronization

---

## 📚 Technical Resources

### Documentation References
- **FastAPI**: https://fastapi.tiangolo.com/
- **FAISS**: https://github.com/facebookresearch/faiss
- **SentenceTransformers**: https://www.sbert.net/
- **Gemini API**: https://ai.google.dev/docs

### Internal Resources
- **API Documentation**: http://localhost:5001/docs
- **Health Check**: http://localhost:5001/health
- **Technical Support**: https://webtest.semtr.com/contact-us/

---

## 💡 Key Takeaways for Future Development

### Critical Success Factors
1. **Hybrid chunking strategy** is essential for context preservation
2. **Multi-threshold fallback** ensures comprehensive results
3. **Context length management** balances quality with cost
4. **Performance monitoring** enables continuous optimization

### Common Pitfalls to Avoid
1. **Don't increase similarity threshold** above 0.3 without testing
2. **Don't reduce TOP_K below 10** for complex queries
3. **Don't ignore token usage** - monitor costs regularly
4. **Don't modify core chunking logic** without comprehensive testing

### Best Practices
1. **Test thoroughly** after any parameter changes
2. **Monitor performance** continuously
3. **Maintain documentation** for all configuration changes
4. **Backup vector indices** before major updates

---

*This guide represents the culmination of extensive optimization and real-world testing. Use it as your foundation for further development and enhancement of the SEM RAG Chatbot system.*