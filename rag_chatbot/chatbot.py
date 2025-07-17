# Main chatbot orchestrator

from dotenv import load_dotenv
load_dotenv()  # Load environment variables

from .document_loader import load_and_chunk_documents
from .embedding import LocalEmbeddingFunction
from .vector_store import FaissVectorStore
from .llm import generate_answer_async
import os
import asyncio

# Configuration constants
EMBEDDING_DIM = 384
VECTOR_STORE_PATH = 'faiss_index.bin'
METADATA_PATH = 'faiss_child_metadata.pkl'
DOCSTORE_PATH = 'faiss_parent_docstore.pkl'

embedding_fn = LocalEmbeddingFunction()
vector_store = FaissVectorStore(EMBEDDING_DIM)

def setup_vector_store():
    if not all(os.path.exists(p) for p in [VECTOR_STORE_PATH, METADATA_PATH, DOCSTORE_PATH]):
        print("INFO: Vector store not found, building a new one...")
        parent_docs, child_docs = load_and_chunk_documents()
        vector_store.add(parent_docs, child_docs, embedding_fn)
        vector_store.save()
        print("INFO: Vector store built and saved successfully.")
    else:
        print("INFO: Loading existing vector store.")
        vector_store.load()

async def get_chatbot_response(user_message):
    # Generate query embedding
    loop = asyncio.get_event_loop()
    query_embedding = await loop.run_in_executor(None, embedding_fn.embed_query, user_message)
    
    # Lower threshold for better case study retrieval
    score_threshold = 0.15
    
    # Run CPU-bound search in thread pool with dynamic parameters
    retrieved_chunks = await loop.run_in_executor(
        None, 
        vector_store.search,
        query_embedding, 
        15,  # top_k - number of chunks to retrieve
        score_threshold,  # similarity threshold
        25000  # max_context_length
    )
    
    # Build context from retrieved chunks
    context = ""
    if retrieved_chunks:
        enriched_chunks = []
        for i, chunk in enumerate(retrieved_chunks):
            enriched_chunk = f"**Information Section {i+1}:**\n{chunk}"
            enriched_chunks.append(enriched_chunk)
        
        context = "\n\n---\n\n".join(enriched_chunks)
        
        # Log token usage estimation
        context_chars = len(context)
        estimated_tokens = int(context_chars * 0.222)
        print(f"--- [TOKEN USAGE] Context: {context_chars} chars, ~{estimated_tokens} tokens")

    # Generate response using LLM
    answer = await generate_answer_async(user_message, context)
    return answer

# Initialize vector store on startup
setup_vector_store()