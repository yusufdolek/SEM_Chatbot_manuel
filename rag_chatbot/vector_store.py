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

    def search(self, query_embedding, top_k=5, score_threshold=0.3):
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)
        
        retrieved_parent_ids = set()
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            score = scores[0][i]
            if score >= score_threshold:
                child_doc = self.metadata[idx]
                parent_id = child_doc.metadata['doc_id']
                retrieved_parent_ids.add(parent_id)

        final_context_docs = [self.docstore[pid] for pid in retrieved_parent_ids]
        
        print(f"--- [VECTOR_STORE DEBUG] Found {len(self.metadata)} child docs. "
              f"Search returned {len(retrieved_parent_ids)} unique parent documents.")
              
        return final_context_docs

# Global değişkene artık ihtiyacımız yok, siliyoruz.