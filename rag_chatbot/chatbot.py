# chatbot.py

from .document_loader import load_and_chunk_documents
from .embedding import LocalEmbeddingFunction
from .vector_store import FaissVectorStore
from .llm import generate_answer
import os

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

def get_chatbot_response(user_message):
    print("Step 1: get_chatbot_response function started.")
    
    print("Step 2: Generating query embedding for the user message...")
    query_embedding = embedding_fn.embed_query(user_message)
    print("Step 2a: Query embedding generated successfully.")
    
    print("Step 3: Searching vector store for relevant documents...")
    retrieved_parent_docs = vector_store.search(
        query_embedding, 
        top_k=10, 
        score_threshold=0.35 
    )
    print("Step 3a: Vector store search completed.")
    
    context = ""
    if retrieved_parent_docs:
        print(f"Step 4: Found {len(retrieved_parent_docs)} relevant parent document(s). Building context string...")
        context = "\n\n---\n\n".join(retrieved_parent_docs)
        print("Step 4a: Context string built successfully.")
    else:
        print("Step 4: No relevant documents found above the threshold.")

    print(f"\n--- CONTEXT TO BE SENT TO LLM ---\n{context[:500]}...\n--------------------------------\n")
    
    print("Step 5: Calling the LLM (generate_answer) to get the final response...")
    answer = generate_answer(user_message, context)
    print("Step 6: Received response from LLM.")
    
    return answer

# Uygulama başladığında vektör veritabanını kur/yükle
setup_vector_store()