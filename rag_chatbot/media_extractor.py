# media_extractor.py

import re
from typing import List, Dict, Tuple

class MediaExtractor:
    """Extracts and processes media URLs from case study content"""
    
    def __init__(self):
        # Common brand name patterns for matching
        self.brand_patterns = {
            'migros': r'\bmigros\b',
            'beymen': r'\bbeymen\b', 
            'boyner': r'\bboyner\b',
            'lc waikiki': r'\b(lc\s*waikiki|lcwaikiki)\b',
            'teknosa': r'\bteknosa\b',
            'turkcell': r'\bturkcell\b',
            'nissan': r'\bnissan\b'
        }
    
    def extract_media_from_content(self, content: str) -> Dict[str, List[str]]:
        """
        Extract image and video URLs from content
        Returns dict with 'images' and 'videos' keys
        """
        media = {'images': [], 'videos': []}
        
        # Extract image URLs (handle multi-line URLs like "https://webtest.semtr.com/wp-\ncontent/uploads/...")
        image_pattern = r'Image URL:\s*(https?://[^\s]+(?:\s*\n\s*[^\s\n]+)*?)(?=\s*\n\s*\n|\s*\n\s*o\s|\s*\n\s*•|\s*$)'
        raw_images = re.findall(image_pattern, content, re.IGNORECASE | re.DOTALL)
        # Clean up multi-line URLs by removing line breaks and extra spaces
        cleaned_images = []
        for url in raw_images:
            if url.strip():
                # Remove all whitespace and newlines from the URL
                clean_url = re.sub(r'\s+', '', url.strip())
                cleaned_images.append(clean_url)
        media['images'] = cleaned_images
        
        # Extract video URLs (handle multi-line URLs)
        video_pattern = r'Video URL:\s*(https?://[^\s]+(?:\s*\n\s*[^\s\n]+)*?)(?=\s*\n\s*\n|\s*\n\s*o\s|\s*\n\s*•|\s*$)'
        raw_videos = re.findall(video_pattern, content, re.IGNORECASE | re.DOTALL)
        # Clean up multi-line URLs by removing line breaks and extra spaces
        cleaned_videos = []
        for url in raw_videos:
            if url.strip():
                # Remove all whitespace and newlines from the URL
                clean_url = re.sub(r'\s+', '', url.strip())
                cleaned_videos.append(clean_url)
        media['videos'] = cleaned_videos
        
        return media
    
    def find_brand_in_query(self, query: str) -> str:
        """
        Find which brand is mentioned in the user query
        Returns normalized brand name or empty string
        """
        query_lower = query.lower()
        
        for brand, pattern in self.brand_patterns.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                return brand
        
        return ""
    
    def extract_case_study_media(self, content: str, query_brand: str = "") -> List[Dict]:
        """
        Extract case studies with their media from content
        Filter by brand if query_brand is provided
        """
        case_studies = []
        
        # Pattern to match case studies with their content
        case_pattern = r'•\s*([^\n]+)\s*\([^)]+\):\s*([^•]*?)(?=\n•|\Z)'
        matches = re.findall(case_pattern, content, re.DOTALL)
        
        for case_name, case_content in matches:
            case_name = case_name.strip()
            
            # If brand filtering is requested, check if this case matches
            if query_brand:
                brand_pattern = self.brand_patterns.get(query_brand, "")
                if brand_pattern and not re.search(brand_pattern, case_name, re.IGNORECASE):
                    continue
            
            # Extract media from this case study
            media = self.extract_media_from_content(case_content)
            
            # Only include cases that have media
            if media['images'] or media['videos']:
                case_studies.append({
                    'name': case_name,
                    'content': case_content.strip(),
                    'media': media
                })
        
        return case_studies
    
    def format_media_markdown(self, media: Dict[str, List[str]], case_name: str = "") -> str:
        """
        Format media URLs as markdown for display
        """
        markdown_parts = []
        
        # Add images
        if media['images']:
            if len(media['images']) == 1:
                markdown_parts.append(f"![{case_name} Case Study]({media['images'][0]})")
            else:
                for i, img_url in enumerate(media['images'], 1):
                    markdown_parts.append(f"![{case_name} Image {i}]({img_url})")
        
        # Add videos (as links since markdown doesn't embed videos directly)
        if media['videos']:
            markdown_parts.append("\n**📺 Related Videos:**")
            for i, video_url in enumerate(media['videos'], 1):
                markdown_parts.append(f"- [View Video {i}]({video_url})")
        
        return "\n\n".join(markdown_parts) if markdown_parts else ""
    
    def extract_special_media(self, content: str, user_query: str) -> str:
        """
        Extract media for special cases like Data Bridge and SmartFeed that aren't in bullet format
        """
        media_markdown = ""
        
        if 'data bridge' in user_query.lower():
            # Find Data Bridge video URL
            video_match = re.search(r'Data Bridge Video URL:\s*(https?://[^\s\n]+)', content, re.IGNORECASE)
            if video_match:
                video_url = video_match.group(1)
                media_markdown += f"**📺 Data Bridge Demo Video:**\n\n[Watch Data Bridge Demo]({video_url})\n\n"
        
        if 'smartfeed' in user_query.lower():
            # Find SmartFeed media
            video_match = re.search(r'SmartFeed Video URL:\s*(https?://[^\s\n]+)', content, re.IGNORECASE)
            image_match = re.search(r'SmartFeed Image URL:\s*(https?://[^\s]+(?:\s*\n\s*[^\s]+)*?)(?=\s|\n\n|$)', content, re.IGNORECASE | re.DOTALL)
            
            if video_match:
                video_url = video_match.group(1)
                media_markdown += f"**📺 SmartFeed Demo Video:**\n\n[Watch SmartFeed Demo]({video_url})\n\n"
            
            if image_match:
                image_url = re.sub(r'\s+', '', image_match.group(1).strip())
                media_markdown += f"![SmartFeed Interface]({image_url})\n\n"
        
        return media_markdown
    
    def enhance_response_with_media(self, response: str, context: str, user_query: str) -> str:
        """
        Main function to enhance LLM response with relevant media
        """
        # Find which brand the user is asking about
        query_brand = self.find_brand_in_query(user_query)
        
        # Extract relevant case studies with media - prioritize brand match
        case_studies = []
        
        if query_brand:
            # First try to find exact brand matches
            case_studies = self.extract_case_study_media(context, query_brand)
            
        # If no brand-specific results, try all case studies
        if not case_studies:
            all_case_studies = self.extract_case_study_media(context)
            
            # If we have a query brand, filter manually for better matching
            if query_brand:
                for cs in all_case_studies:
                    brand_pattern = self.brand_patterns.get(query_brand, "")
                    if brand_pattern and re.search(brand_pattern, cs['name'], re.IGNORECASE):
                        case_studies.append(cs)
                
                # If still no matches, take all (fallback)
                if not case_studies:
                    case_studies = all_case_studies
            else:
                case_studies = all_case_studies
        
        # Handle special cases like Data Bridge and SmartFeed
        if 'data bridge' in user_query.lower() or 'smartfeed' in user_query.lower():
            # These aren't in bullet format, extract directly
            special_media = self.extract_special_media(context, user_query)
            if special_media:
                response += "\n\n---\n\n" + special_media
                return response
        
        # If we found relevant media, append it to the response
        if case_studies:
            media_section = "\n\n---\n\n"
            
            # Add media for each relevant case study (limit based on brand match)
            max_cases = 1 if query_brand else 3  # Show only 1 if brand-specific query
            for case_study in case_studies[:max_cases]:
                media_markdown = self.format_media_markdown(
                    case_study['media'], 
                    case_study['name']
                )
                if media_markdown:
                    media_section += f"**{case_study['name']} Visual:**\n\n{media_markdown}\n\n"
            
            # Only append if we actually found media
            if len(media_section) > 20:  # More than just separators
                response += media_section
        
        return response