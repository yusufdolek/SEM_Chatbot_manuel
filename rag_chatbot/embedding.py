from sentence_transformers import SentenceTransformer
import re

MODEL_NAME = "all-MiniLM-L6-v2"

class LocalEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        
        # Brand name mappings for better semantic matching
        self.brand_mappings = {
            'lcwaikiki': 'LC WAIKIKI',
            'lc waikiki': 'LC WAIKIKI', 
            'beymen': 'BEYMEN',
            'migros': 'MIGROS',
            'boyner': 'BOYNER',
            'teknosa': 'TEKNOSA',
            'turkcell': 'TURKCELL',
            'nissan': 'NISSAN',
            'burger king': 'BURGER KING',
            'turkiye is bankasi': 'TÜRKİYE İŞ BANKASI',
            'qnb finansbank': 'QNB FINANSBANK'
        }
        
        # Synonyms for project/case study terms
        self.project_synonyms = {
            'project': ['project', 'case study', 'success story', 'campaign', 'case', 'work'],
            'case study': ['project', 'case study', 'success story', 'campaign', 'case', 'work'],
            'campaign': ['project', 'case study', 'success story', 'campaign', 'case', 'work']
        }
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing for better semantic matching"""
        if not text:
            return text
            
        # Convert to lowercase for processing
        processed = text.lower()
        
        # Normalize brand names
        for brand_key, brand_value in self.brand_mappings.items():
            processed = re.sub(r'\b' + re.escape(brand_key) + r'\b', brand_value, processed, flags=re.IGNORECASE)
        
        # Add project/case study synonyms for query expansion
        for term, synonyms in self.project_synonyms.items():
            if term in processed:
                # Add synonyms to increase matching probability
                synonym_text = ' '.join(synonyms)
                processed = f"{processed} {synonym_text}"
                break
        
        return processed

    def embed_documents(self, texts):
        # Preprocess documents for better semantic matching
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.model.encode(processed_texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        # Preprocess query for better semantic matching
        processed_text = self.preprocess_text(text)
        return self.model.encode([processed_text])[0].tolist()