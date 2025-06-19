# vector_store.py

import faiss
import numpy as np
import os
import pickle

VECTOR_STORE_PATH = 'faiss_index.bin'
METADATA_PATH = 'faiss_metadata.pkl'

class FaissVectorStore:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        # Kosinüs benzerliği için İç Çarpım endeksini kullan
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []

    def add_embeddings(self, embeddings, metadatas_to_store):
        np_embeddings = np.array(embeddings).astype('float32')
        # IndexFlatIP için vektörleri normalize et
        faiss.normalize_L2(np_embeddings)
        self.index.add(np_embeddings)
        self.metadata.extend(metadatas_to_store)

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

    def search(self, query_embedding, top_k=5, score_threshold=0.3):
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_embedding]).astype('float32')
        # Sorgu vektörünü de normalize et
        faiss.normalize_L2(query_vector)

        # Artık 'scores' doğrudan kosinüs benzerliği puanıdır (0-1 arası)
        scores, indices = self.index.search(query_vector, top_k)
        print("\n--- [VECTOR_STORE DEBUG] Raw Search Results (Before Threshold) ---")
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            
            score = scores[0][i]

            print(f"  Chunk Index: {idx}, Raw Score: {score:.4f}")

            if score >= score_threshold:
                retrieved_item = self.metadata[idx]
                results.append((retrieved_item, score))

        print(f"--- [VECTOR_STORE DEBUG] Found {len(results)} chunks above threshold {score_threshold} ---")
        return results