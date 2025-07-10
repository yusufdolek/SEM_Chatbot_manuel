# media_mapping.py
# Simple, maintainable media URL mapping system

import re
from typing import Dict, List

class MediaMapping:
    """Simple media URL mapping for services and case studies"""
    
    def __init__(self):
        # Static media mapping - easy to maintain and extend
        self.media_map = {
            # Google Services
            'google analytics 360': {
                'images': [],
                'videos': ['https://www.youtube.com/watch?v=9Y4U7REuHcg']
            },
            'search ads 360': {
                'images': ['https://improvado.io/5a1eb87c9afe1000014a4c7d/64e351decceb1eb3cec39ac3_5cb03a81fbe81c038054e534_gmp_search_ads_360_90.png'],
                'videos': []
            },
            'campaign manager 360': {
                'images': ['https://ppcexpo.com/blog/wp-content/uploads/2024/10/google-campaign-manager-360-1-1.jpg'],
                'videos': []
            },
            'display video 360': {
                'images': [],
                'videos': ['https://www.youtube.com/watch?app=desktop&v=ISB-KOW3oCI']
            },
            'google meridian': {
                'images': [],
                'videos': ['https://www.youtube.com/watch?v=5ag97Phtw4Y']
            },
            'cltv': {
                'images': [],
                'videos': ['https://www.youtube.com/watch?v=kinhxJvA4a0']
            },
            'customer lifetime value': {
                'images': [],
                'videos': ['https://www.youtube.com/watch?v=kinhxJvA4a0']
            },
            
            # SEM Products
            'smartfeed': {
                'images': ['https://webtest.semtr.com/wp-content/uploads/2025/05/Hero-header-section-1.png'],
                'videos': ['https://www.youtube.com/watch?v=hA3o8C8P71o']
            },
            'data bridge': {
                'images': [],
                'videos': ['https://www.youtube.com/watch?v=KeCaFyQLFV8']
            },
            
            # Company Info
            'sem journey': {
                'images': ['https://webtest.semtr.com/wp-content/uploads/2025/05/c3e0e556c647c3c8d815e5bef5878c33.png'],
                'videos': []
            },
            'sem milestones': {
                'images': ['https://webtest.semtr.com/wp-content/uploads/2025/05/c3e0e556c647c3c8d815e5bef5878c33.png'],
                'videos': []
            },
            
            # Brand Case Studies
            'migros': {
                'images': ['https://webtest.semtr.com/wp-content/uploads/2025/05/25ce6a54bc5e301bb7dd9aff52bf8e33.png'],
                'videos': []
            },
            'beymen': {
                'images': ['https://webtest.semtr.com/wp-content/uploads/2025/05/Beymen.com_.png'],
                'videos': []
            },
            'boyner': {
                'images': ['https://webtest.semtr.com/wp-content/uploads/2025/05/c279dd2eabd8ce7c0fcf30accd80aacb.svg'],
                'videos': []
            },
            'lc waikiki': {
                'images': ['https://webtest.semtr.com/wp-content/uploads/2025/05/a0a21448ddc2f13b87f5dcf9e012a430.png'],
                'videos': []
            },
            'popeyes': {
                'images': ['https://webtest.semtr.com/wp-content/uploads/2025/05/5002ccbfed36ca9c3eb08579516ab5c6.png'],
                'videos': []
            },
            'tab gıda': {
                'images': ['https://webtest.semtr.com/wp-content/uploads/2025/05/747f87fce92bc453444c56d049bf0739.png'],
                'videos': []
            },
            'dominos': {
                'images': [],
                'videos': ['https://www.youtube.com/watch?v=-BuCQ47DWck']
            },
            'bilyoner': {
                'images': ['https://webtest.semtr.com/wp-content/uploads/2025/05/748fe9386257b4c999fe28e59561ad01.png'],
                'videos': []
            }
        }
        
        # Keywords that should trigger media display
        self.media_keywords = [
            'smartfeed', 'data bridge', 'case study', 'başarı hikayesi', 'örnek proje',
            'google analytics 360', 'search ads 360', 'campaign manager 360', 'display video 360',
            'google meridian', 'cltv', 'customer lifetime value',
            'görsel', 'resim', 'image', 'video', 'demo', 'screenshot'
        ]
        
        # Brand keywords
        self.brand_keywords = list(self.media_map.keys())
    
    def load_media_from_pdf(self, pdf_content: str):
        """Load media URLs from PDF content - already implemented in media_map above"""
        # All media URLs from PDF have been loaded into the media_map dictionary
        # This function is kept for future PDF updates if needed
        pass
    
    def find_relevant_media(self, user_query: str) -> Dict[str, List[str]]:
        """Find relevant media for a user query"""
        query_lower = user_query.lower()
        
        # Check for exact matches first
        for key, media in self.media_map.items():
            if key in query_lower:
                if media['images'] or media['videos']:
                    return {
                        'key': key,
                        'images': media['images'],
                        'videos': media['videos']
                    }
        
        # For partial matches, require more specific matching for Google services
        for key, media in self.media_map.items():
            # Special handling for Google services - require "360" for specific products
            if 'google' in key and '360' in key:
                if key.replace(' ', '').replace('360', '') in query_lower.replace(' ', '') and '360' in query_lower:
                    if media['images'] or media['videos']:
                        return {
                            'key': key,
                            'images': media['images'],
                            'videos': media['videos']
                        }
            # For other services, use normal partial matching
            elif 'google' not in key or '360' not in key:
                key_words = key.split()
                if any(word in query_lower for word in key_words if len(word) > 3):
                    if media['images'] or media['videos']:
                        return {
                            'key': key,
                            'images': media['images'],
                            'videos': media['videos']
                        }
        
        return {'key': '', 'images': [], 'videos': []}
    
    def should_show_media(self, user_query: str) -> bool:
        """Determine if query should show media"""
        query_lower = user_query.lower()
        
        # Always show media for specific keywords
        if any(keyword in query_lower for keyword in self.media_keywords):
            return True
            
        # Always show media for brand mentions
        if any(brand in query_lower for brand in self.brand_keywords):
            return True
        
        # Don't show media for general questions
        general_keywords = [
            'merhaba', 'hello', 'sem ne yapar', 'what does sem do', 'hakkında', 'about',
            'nasıl', 'how', 'ne yapıyor', 'what do you do'
        ]
        
        if any(keyword in query_lower for keyword in general_keywords):
            return False
            
        return False
    
    def youtube_url_to_embed(self, youtube_url: str) -> str:
        """Convert YouTube URL to embeddable format"""
        
        # Handle different YouTube URL formats
        if 'youtube.com/watch' in youtube_url:
            # Extract video ID from various watch URL formats
            match = re.search(r'[?&]v=([^&]+)', youtube_url)
            if match:
                video_id = match.group(1)
            else:
                return youtube_url
        elif 'youtu.be/' in youtube_url:
            video_id = youtube_url.split('youtu.be/')[1].split('?')[0]
        elif 'youtube.com/embed/' in youtube_url:
            return youtube_url  # Already embed format
        else:
            return youtube_url  # Return original if can't parse
            
        return f"https://www.youtube.com/embed/{video_id}"

    def format_media_html(self, media: Dict[str, List[str]], title: str = "") -> str:
        """Format media as HTML for proper sizing and video embedding"""
        if not media['images'] and not media['videos']:
            return ""
            
        html_parts = []
        
        # Add images with proper sizing and click functionality
        if media['images']:
            for i, img_url in enumerate(media['images'], 1):
                alt_text = f"{title} Image {i}" if len(media['images']) > 1 else f"{title} Interface"
                html_parts.append(f'<img src="{img_url}" alt="{alt_text}" class="clickable-image" style="width: 300px; height: auto; max-width: 100%; border-radius: 8px; margin: 10px 0; display: block; cursor: pointer;" title="Click to view full size">')
        
        # Add videos as embeddable iframes
        if media['videos']:
            for i, video_url in enumerate(media['videos'], 1):
                embed_url = self.youtube_url_to_embed(video_url)
                html_parts.append(f'<iframe src="{embed_url}" style="width: 300px; height: 169px; max-width: 100%; border-radius: 8px; margin: 10px 0; border: none;" allowfullscreen></iframe>')
        
        return "\n\n".join(html_parts)
    
    def enhance_response(self, response: str, user_query: str) -> str:
        """Main function to enhance response with media"""
        if not self.should_show_media(user_query):
            return response
            
        media = self.find_relevant_media(user_query)
        if not media['images'] and not media['videos']:
            return response
            
        media_html = self.format_media_html(media, media['key'].title())
        if media_html:
            return response + f"\n\n---\n\n{media_html}\n\n"
            
        return response