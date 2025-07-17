from sentence_transformers import SentenceTransformer
import re

MODEL_NAME = "all-MiniLM-L6-v2"

class LocalEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        
        # Simple brand name mappings for consistency
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
    
    def preprocess_text(self, text):
        """Simple text preprocessing for brand normalization"""
        if not text:
            return text
            
        # Convert to lowercase for processing
        processed = text.lower()
        
        # Simple brand name normalization
        for brand_key, brand_value in self.brand_mappings.items():
            processed = re.sub(r'\b' + re.escape(brand_key) + r'\b', brand_value, processed, flags=re.IGNORECASE)
        
        return processed

    def embed_documents(self, texts):
        # Preprocess documents for better semantic matching
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.model.encode(processed_texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        # Preprocess query for better semantic matching
        processed_text = self.preprocess_text(text)
        return self.model.encode([processed_text])[0].tolist()