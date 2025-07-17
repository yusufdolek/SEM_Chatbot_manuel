# vector_store.py

import faiss
import numpy as np
import os
import pickle

VECTOR_STORE_PATH = 'faiss_index.bin'
METADATA_PATH = 'faiss_child_metadata.pkl'
DOCSTORE_PATH = 'faiss_parent_docstore.pkl'

class FaissVectorStore:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.docstore = {} 
        self.metadata = [] 

    # 'embedding_fn' ARTIK BİR PARAMETRE
    def add(self, parent_docs, child_docs, embedding_fn):
        """
        Parent ve Child dokümanları alır, child'ları vektörleştirir ve depolar.
        """
        self.docstore = {doc.metadata['doc_id']: doc.page_content for doc in parent_docs}
        
        texts_for_embedding = [doc.page_content for doc in child_docs]
        
        # PARAMETRE OLARAK GELEN embedding_fn'i KULLANIYORUZ
        np_embeddings = np.array(embedding_fn.embed_documents(texts_for_embedding)).astype('float32')
        
        if np_embeddings.shape[0] > 0:
            faiss.normalize_L2(np_embeddings)
            self.index.add(np_embeddings)
            self.metadata.extend(child_docs)

    def save(self):
        faiss.write_index(self.index, VECTOR_STORE_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)
        with open(DOCSTORE_PATH, 'wb') as f:
            pickle.dump(self.docstore, f)

    # load METODUNUN ARTIK embedding_fn PARAMETRESİNE İHTİYACI YOK
    def load(self):
        if os.path.exists(VECTOR_STORE_PATH):
            self.index = faiss.read_index(VECTOR_STORE_PATH)
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
        if os.path.exists(DOCSTORE_PATH):
            with open(DOCSTORE_PATH, 'rb') as f:
                self.docstore = pickle.load(f)

    def search(self, query_embedding, top_k=8, score_threshold=0.35, max_context_length=25000):
        """
        OPTIMIZED VERSION: Returns only relevant child chunks instead of full parent documents
        - Higher similarity threshold (0.4) for better quality
        - More results (top_k=8) for detailed responses
        - Context length limit (25000) for comprehensive answers
        """
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)
        
        relevant_chunks = []
        total_context_length = 0
        
        for i, idx in enumerate(indices[0]):
            if idx == -1: 
                continue
            
            score = scores[0][i]
            if score >= score_threshold:
                child_doc = self.metadata[idx]
                chunk_content = child_doc.page_content
                chunk_length = len(chunk_content)
                
                # Check if adding this chunk would exceed context limit
                if total_context_length + chunk_length > max_context_length:
                    print(f"--- [CONTEXT LIMIT] Stopping at {len(relevant_chunks)} chunks")
                    break
                
                relevant_chunks.append(chunk_content)
                total_context_length += chunk_length
        
        # If no results above threshold, try with lower threshold as fallback
        if not relevant_chunks and score_threshold > 0.2:
            print(f"--- [FALLBACK] No results with threshold {score_threshold}, trying 0.2")
            return self.search(query_embedding, top_k, 0.2, max_context_length)
        
        print(f"--- [VECTOR_STORE] Found {len(self.metadata)} child docs. "
              f"Returned {len(relevant_chunks)} relevant chunks ({total_context_length} chars total).")
              
        return relevant_chunks

# Global değişkene artık ihtiyacımız yok, siliyoruz.