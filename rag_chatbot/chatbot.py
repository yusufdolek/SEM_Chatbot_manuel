from .document_loader import load_and_chunk_documents
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
    if not (os.path.exists('faiss_index.bin') and os.path.exists('faiss_metadata.pkl') and False):
        print("INFO:Vector store not found. Building a new one...") #
        # docs = load_documents()
        chunks = load_and_chunk_documents()
        
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
    
    top_chunks_with_scores = vector_store.search(
        query_embedding, 
        top_k=5, 
        score_threshold=0.3
    )
    
    context = ""

    if top_chunks_with_scores:
        print(f"INFO: Found {len(top_chunks_with_scores)} relevant chunks. Building context...")
        
        context_parts = []
        
        # --- DÖNGÜYÜ DOĞRU ŞEKİLDE KURUYORUZ ---
        # Her bir elemanı (chunk_data, score) olarak ikiye ayırıyoruz.
        for chunk_data, score in top_chunks_with_scores:
            # chunk_data artık {"page_content": "...", "metadata": {...}} şeklinde bir sözlük
            metadata = chunk_data.get('metadata', {})
            page_content = chunk_data.get('page_content', '')
            
            header_info = ""
            if 'Header 1' in metadata:
                header_info += f"From Section: {metadata['Header 1']}"
            if 'Header 2' in metadata:
                header_info += f" > {metadata['Header 2']}"
            if 'Header 3' in metadata:
                header_info += f" > {metadata['Header 3']}"
            
            # (İsteğe bağlı) Hata ayıklama için puanı da ekleyebiliriz
            # header_info += f" (Relevance: {score:.2f})"
            
            if header_info:
                context_parts.append(f"{header_info}\n---\n{page_content}")
            else:
                context_parts.append(page_content)

        context = '\n\n'.join(context_parts)
    
    else:
        print("INFO: No relevant documents found. Sending empty context to LLM.")

    # print(f"--- CONTEXT SENT TO LLM ---\n{context}\n--------------------------") # Gerekirse hata ayıklama için
    
    answer = generate_answer(user_message, context)
    
    return answer


setup_vector_store()