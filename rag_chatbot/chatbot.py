from .document_loader import load_documents, chunk_documents
from .embedding import LocalEmbeddingFunction
from .vector_store import FaissVectorStore
from .llm import generate_answer
import os

# On first run, build the vector store
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 embedding size
embedding_fn = LocalEmbeddingFunction()
vector_store = FaissVectorStore(EMBEDDING_DIM)

# Check if vector store exists, else build it
def setup_vector_store():
    if not (os.path.exists('faiss_index.bin') and os.path.exists('faiss_metadata.pkl')):
        print("INFO:Vector store not found. Building a new one...") #
        docs = load_documents()
        chunks = chunk_documents(docs)
        
        # 2. Embedding için metinleri ve saklamak için metadataları hazırla
        texts_for_embedding = [chunk.page_content for chunk in chunks]
        metadata_to_store = [
            {"page_content": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ]

        # 3. Metinleri vektörlere dönüştür
        embeddings = embedding_fn.embed_documents(texts_for_embedding)
        

        vector_store.add_embeddings(embeddings, metadata_to_store)
        vector_store.save()
        print("INFO: Vector store built and saved successfully.")
    else:
        print("INFO: Loading existing vector store.") #
        vector_store.load()

def get_chatbot_response(user_message):
    query_embedding = embedding_fn.embed_query(user_message)
    top_chunks_content = vector_store.search(query_embedding, top_k=10)
    
    # --- HATA AYIKLAMA İÇİN BU SATIRLARI EKLEYİN ---
    print("="*50)
    print(f"KULLANICI SORUSU: {user_message}")
    print("--- LLM'e GÖNDERİLEN İLK 10 CHUNK ---")
    for i, chunk in enumerate(top_chunks_content):
        print(f"--- CHUNK {i+1} ---")
        print(chunk)
    print("="*50)
    # --- HATA AYIKLAMA BİTTİ ---

    context = '\n\n---\n\n'.join(top_chunks_content)
    answer = generate_answer(user_message, context)
    return answer


setup_vector_store()