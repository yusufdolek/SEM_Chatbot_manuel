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
    # Run CPU-bound search in thread pool
    retrieved_parent_docs = await loop.run_in_executor(
        None, 
        vector_store.search,
        query_embedding, 
        10,  # top_k
        0.25  # score_threshold
    )
    
    # Check if we have detailed case studies document
    has_detailed_cases = any('Formula-Based Bidding' in doc or 'Mid-Funnel Growth' in doc for doc in retrieved_parent_docs)
    
    # For brand-specific queries, ensure we get the detailed case studies document
    brand_keywords = ['beymen', 'migros', 'boyner', 'lc waikiki', 'lcwaikiki']
    is_brand_query = any(brand in user_message.lower() for brand in brand_keywords)
    
    if is_brand_query and not has_detailed_cases:
        # Try specific queries to get the detailed case studies document
        enhanced_queries = [
            f"{user_message} formula based bidding search ads",
            f"{user_message} mid funnel growth demand gen",
            "case studies success stories with image URL",
            "projects formula based bidding image"
        ]
        
        for enhanced_query in enhanced_queries:
            enhanced_embedding = await loop.run_in_executor(None, embedding_fn.embed_query, enhanced_query)
            enhanced_docs = await loop.run_in_executor(
                None,
                vector_store.search,
                enhanced_embedding,
                5,  # top_k
                0.15  # score_threshold
            )
            
            # Check if these docs have detailed case studies
            has_detailed = any('Formula-Based Bidding' in doc or 'Mid-Funnel Growth' in doc for doc in enhanced_docs)
            if has_detailed:
                print(f"--- [ENHANCED SEARCH] Found detailed case studies with query: {enhanced_query}")
                retrieved_parent_docs = enhanced_docs
                break
    
    # If still no detailed cases, try one more fallback for non-brand queries
    elif not has_detailed_cases:
        enhanced_queries = [
            f"{user_message} case study with image",
            f"{user_message} success story"
        ]
        
        for enhanced_query in enhanced_queries:
            enhanced_embedding = await loop.run_in_executor(None, embedding_fn.embed_query, enhanced_query)
            enhanced_docs = await loop.run_in_executor(
                None,
                vector_store.search,
                enhanced_embedding,
                5,  # top_k
                0.2  # score_threshold
            )
            
            # Check if these docs have what we need
            has_cases = any('Image URL:' in doc for doc in enhanced_docs)
            if has_cases:
                print(f"--- [ENHANCED SEARCH] Found better documents with enhanced query: {enhanced_query}")
                retrieved_parent_docs = enhanced_docs
                break
    print("Step 3a: Vector store search completed.")
    
    context = ""
    if retrieved_parent_docs:
        print(f"Step 4: Found {len(retrieved_parent_docs)} relevant parent document(s). Building context string...")
        
        # Prioritize documents with detailed case studies for brand queries
        if is_brand_query:
            # Sort documents: detailed case studies first, others after
            detailed_docs = []
            other_docs = []
            
            for doc in retrieved_parent_docs:
                has_detailed_cases = 'Formula-Based Bidding' in doc or 'Mid-Funnel Growth' in doc
                if has_detailed_cases:
                    detailed_docs.append(doc)
                else:
                    other_docs.append(doc)
            
            # Put detailed docs first
            prioritized_docs = detailed_docs + other_docs
            print(f"--- [PRIORITIZATION] Moved {len(detailed_docs)} detailed case study documents to front")
        else:
            prioritized_docs = retrieved_parent_docs
        
        context = "\n\n---\n\n".join(prioritized_docs)
        print("Step 4a: Context string built successfully.")
    else:
        print("Step 4: No relevant documents found above the threshold.")

    print(f"\n--- CONTEXT TO BE SENT TO LLM ---\n{context[:500]}...\n--------------------------------\n")
    
    print("Step 5: Calling the async LLM (generate_answer_async) to get the final response...")
    answer = await generate_answer_async(user_message, context)
    print("Step 6: Received response from async LLM.")
    
    return answer

# Uygulama başladığında vektör veritabanını kur/yükle
setup_vector_store()