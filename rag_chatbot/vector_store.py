# vector_store.py DOSYASINDAKİ DOĞRU KOD

import faiss
import numpy as np
import os
import pickle

VECTOR_STORE_PATH = 'faiss_index.bin'
METADATA_PATH = 'faiss_metadata.pkl'

class FaissVectorStore:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = [] # Bu satırı bu şekilde bırakalım

    # --- BU FONKSİYONU AŞAĞIDAKİ GİBİ GÜNCELLEYİN ---
    def add_embeddings(self, embeddings, metadatas_to_store): # İkinci parametrenin adını değiştirelim ki karışmasın
        np_embeddings = np.array(embeddings).astype('float32')
        self.index.add(np_embeddings)
        self.metadata.extend(metadatas_to_store) # extend() metodu ile listeye ekleme yapalım

    def save(self):
        faiss.write_index(self.index, VECTOR_STORE_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(VECTOR_STORE_PATH):
            self.index = faiss.read_index(VECTOR_STORE_PATH)
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'rb') as f:
                self.metadata = pickle.load(f)

    # --- BU FONKSİYONUN DA DOĞRU OLDUĞUNDAN EMİN OLUN ---
    # --- MEVCUT `search` FONKSİYONUNU BU YENİ VERSİYONLA DEĞİŞTİRİN ---
    def search(self, query_embedding, top_k=5, score_threshold=0.7):
        if self.index.ntotal == 0:
            return []

        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        
        # --- HATA AYIKLAMA İÇİN BU SATIRI EKLEYİN ---
        print("--- DEBUG: Raw Scores ---")
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            score = 1.0 / (1.0 + distances[0][i])
            
            # --- HATA AYIKLAMA İÇİN BU SATIRI EKLEYİN ---
            print(f"Chunk {idx}: Score = {score:.4f}")
            
            if score >= score_threshold:
                retrieved_item = self.metadata[idx]
                results.append((retrieved_item, score))
        
        # --- HATA AYIKLAMA İÇİN BU SATIRI EKLEYİN ---
        print(f"--- Found {len(results)} chunks above threshold {score_threshold} ---")
        
        return results