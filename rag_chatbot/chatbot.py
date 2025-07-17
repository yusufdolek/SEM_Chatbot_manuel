# chatbot.py

from dotenv import load_dotenv
load_dotenv()  # Load environment variables first

from .document_loader import load_and_chunk_documents
from .embedding import LocalEmbeddingFunction
from .vector_store import FaissVectorStore
from .llm import generate_answer_async
import os
import asyncio

# Değişkenleri en üste, daha okunaklı bir yere taşıyoruz
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
    print("Step 1: get_chatbot_response function started.")
    
    print("Step 2: Generating query embedding for the user message...")
    # Run CPU-bound embedding in thread pool
    loop = asyncio.get_event_loop()
    query_embedding = await loop.run_in_executor(None, embedding_fn.embed_query, user_message)
    print("Step 2a: Query embedding generated successfully.")
    
    print("Step 3: Searching vector store for relevant documents...")
    
    # Dynamic threshold based on query length and complexity
    query_length = len(user_message)
    if query_length > 20:
        # Longer queries get lower threshold for more comprehensive results
        dynamic_threshold = 0.3
        print(f"--- [DYNAMIC THRESHOLD] Long query ({query_length} chars) - using threshold {dynamic_threshold}")
    elif query_length < 10:
        # Short queries get higher threshold for more focused results
        dynamic_threshold = 0.4
        print(f"--- [DYNAMIC THRESHOLD] Short query ({query_length} chars) - using threshold {dynamic_threshold}")
    else:
        # Medium queries get balanced threshold
        dynamic_threshold = 0.35
        print(f"--- [DYNAMIC THRESHOLD] Medium query ({query_length} chars) - using threshold {dynamic_threshold}")
    
    # Run CPU-bound search in thread pool with dynamic parameters
    retrieved_chunks = await loop.run_in_executor(
        None, 
        vector_store.search,
        query_embedding, 
        8,  # top_k - increased for detailed responses
        dynamic_threshold,  # dynamic score_threshold based on query
        25000  # max_context_length - increased for comprehensive answers
    )
    
    # Check if we have detailed case studies in retrieved chunks
    has_detailed_cases = any('Formula-Based Bidding' in chunk or 'Mid-Funnel Growth' in chunk for chunk in retrieved_chunks)
    
    # For brand-specific queries, ensure we get the detailed case studies
    brand_keywords = ['beymen', 'migros', 'boyner', 'lc waikiki', 'lcwaikiki']
    is_brand_query = any(brand in user_message.lower() for brand in brand_keywords)
    
    if is_brand_query and not has_detailed_cases:
        # Try specific queries to get the detailed case studies
        enhanced_queries = [
            f"{user_message} formula based bidding search ads",
            f"{user_message} mid funnel growth demand gen",
            "case studies success stories with image URL",
            "projects formula based bidding image"
        ]
        
        for enhanced_query in enhanced_queries:
            enhanced_embedding = await loop.run_in_executor(None, embedding_fn.embed_query, enhanced_query)
            enhanced_chunks = await loop.run_in_executor(
                None,
                vector_store.search,
                enhanced_embedding,
                8,  # top_k
                0.2,  # score_threshold - slightly lower for fallback
                25000  # max_context_length
            )
            
            # Check if these chunks have detailed case studies
            has_detailed = any('Formula-Based Bidding' in chunk or 'Mid-Funnel Growth' in chunk for chunk in enhanced_chunks)
            if has_detailed:
                print(f"--- [ENHANCED SEARCH] Found detailed case studies with query: {enhanced_query}")
                retrieved_chunks = enhanced_chunks
                break
    
    # If still no detailed cases, try one more fallback for non-brand queries
    elif not has_detailed_cases:
        enhanced_queries = [
            f"{user_message} case study with image",
            f"{user_message} success story"
        ]
        
        for enhanced_query in enhanced_queries:
            enhanced_embedding = await loop.run_in_executor(None, embedding_fn.embed_query, enhanced_query)
            enhanced_chunks = await loop.run_in_executor(
                None,
                vector_store.search,
                enhanced_embedding,
                8,  # top_k
                0.25,  # score_threshold
                25000  # max_context_length
            )
            
            # Check if these chunks have what we need
            has_cases = any('Image URL:' in chunk for chunk in enhanced_chunks)
            if has_cases:
                print(f"--- [ENHANCED SEARCH] Found better documents with enhanced query: {enhanced_query}")
                retrieved_chunks = enhanced_chunks
                break
    print("Step 3a: Vector store search completed.")
    
    context = ""
    if retrieved_chunks:
        print(f"Step 4: Found {len(retrieved_chunks)} relevant chunk(s). Building context string...")
        
        # Prioritize chunks with detailed case studies for brand queries
        if is_brand_query:
            # Sort chunks: detailed case studies first, others after
            detailed_chunks = []
            other_chunks = []
            
            for chunk in retrieved_chunks:
                has_detailed_cases = 'Formula-Based Bidding' in chunk or 'Mid-Funnel Growth' in chunk
                if has_detailed_cases:
                    detailed_chunks.append(chunk)
                else:
                    other_chunks.append(chunk)
            
            # Put detailed chunks first
            prioritized_chunks = detailed_chunks + other_chunks
            print(f"--- [PRIORITIZATION] Moved {len(detailed_chunks)} detailed case study chunks to front")
        else:
            prioritized_chunks = retrieved_chunks
        
        # Enrich context with metadata and structure
        enriched_chunks = []
        for i, chunk in enumerate(prioritized_chunks):
            # Add chunk numbering and structure
            enriched_chunk = f"**Information Section {i+1}:**\n{chunk}"
            enriched_chunks.append(enriched_chunk)
        
        context = "\n\n---\n\n".join(enriched_chunks)
        print("Step 4a: Context string built successfully.")
        
        # Log token usage estimation
        context_chars = len(context)
        estimated_tokens = int(context_chars * 0.222)
        print(f"--- [TOKEN USAGE] Context: {context_chars} chars, ~{estimated_tokens} tokens")
    else:
        print("Step 4: No relevant chunks found above the threshold.")

    print(f"\n--- CONTEXT TO BE SENT TO LLM ---\n{context[:500]}...\n--------------------------------\n")
    
    print("Step 5: Calling the async LLM (generate_answer_async) to get the final response...")
    answer = await generate_answer_async(user_message, context)
    print("Step 6: Received response from async LLM.")
    
    return answer

# Uygulama başladığında vektör veritabanını kur/yükle
setup_vector_store()