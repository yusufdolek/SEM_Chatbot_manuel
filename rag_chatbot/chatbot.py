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
    if not (os.path.exists('faiss_index.bin') and os.path.exists('faiss_metadata.pkl')):
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
    
    # search fonksiyonu artık bize {"page_content": ..., "metadata": ...} döndürmeli
    # vector_store.py'deki search fonksiyonunu buna göre güncelleyin.
    # Önceki adımdaki gibi 'page_content' döndürmek yerine tüm nesneyi döndürsün.
    top_chunks = vector_store.search(query_embedding, top_k=3) # Yeni bir fonksiyon yazmak daha temiz olur

    # CONTEXT'İ AKILLICA OLUŞTURMA
    context_parts = []
    for chunk in top_chunks:
        # chunk artık {"page_content": "...", "metadata": {"source": "...", "Header 1": "..."}} formatında
        metadata = chunk.get('metadata', {}) # Hata almamak için .get() kullanalım
        page_content = chunk.get('page_content', '')
        
        header_info = ""
        # Metadata içindeki başlıkları hiyerarşik olarak birleştir
        if 'Header 1' in metadata:
            header_info += f"From Section: {metadata['Header 1']}"
        if 'Header 2' in metadata:
            header_info += f" > {metadata['Header 2']}"
        if 'Header 3' in metadata:
            header_info += f" > {metadata['Header 3']}"
        
        # Eğer başlık bilgisi varsa, içeriğin başına ekle
        if header_info:
            context_parts.append(f"{header_info}\n---\n{page_content}")
        else:
            context_parts.append(page_content)

    context = '\n\n'.join(context_parts)
    
    # Hata ayıklama için context'i yazdırabilirsiniz:
    # print(context)
    
    answer = generate_answer(user_message, context)
    return answer


setup_vector_store()