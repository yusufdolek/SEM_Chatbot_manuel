# media_extractor.py

from .media_mapping import MediaMapping

class MediaExtractor:
    """Simple media extractor using MediaMapping system"""
    
    def __init__(self):
        self.media_mapping = MediaMapping()
    
    def enhance_response_with_media(self, response: str, context: str, user_query: str) -> str:
        """
        Main function to enhance LLM response with relevant media
        Now uses the simplified MediaMapping system
        """
        # Context parameter kept for API compatibility but not used in new system
        return self.media_mapping.enhance_response(response, user_query)