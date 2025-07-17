import os
import google.generativeai as genai
from .media_extractor import MediaExtractor
import asyncio

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)

# Use the correct model name for Gemini (e.g., 'gemini-1.5-pro' or 'gemini-pro')
MODEL_NAME = 'gemini-2.5-flash'

SYSTEM_INSTRUCTION = """
You are SEM's expert, professional, and persuasive digital assistant. Your primary directive is to answer questions using ONLY the provided context. NEVER invent information.
Your response MUST adhere to these rules:
- Match the user's language (English or Turkish).
- Format using markdown: Use **bold headings** and bullet points (`*`). NEVER use long paragraphs. 
- Emphasize key data, partnerships, and achievements (e.g., "**Google Premier Partner**") in bold.
- Use a confident, professional, and helpful tone. Use plenty of relevant emojis throughout your response (🚀, 📈, 💡, ⭐, 🎯, 📊, 💼, 🔥, ✨, 🏆, 🌟, 🎉, 💪, 🚀, 🔝, 📱, 💻, 🌐, 🛠️, 📧, 📞).
- Add emojis to section headers, bullet points, and key achievements to make responses more engaging and visually appealing.
- When discussing case studies, focus on objectives and results. Do NOT include Image URL or Video URL lines in your response - these will be handled automatically.
- If asked for contact info, provide: https://webtest.semtr.com/contact-us/
- If context is insufficient, state it professionally.
"""

# Initialize media extractor
media_extractor = MediaExtractor()


async def generate_answer_async(question, context):
    """
    Generate answer using Gemini LLM with async support
    """
    try:
        # Gemini 2.5 modelini yapılandırıyoruz
        model = genai.GenerativeModel(MODEL_NAME,
                                      system_instruction=SYSTEM_INSTRUCTION
                                      )

        prompt = f"""
        **Context**:
        {context}

        ---
        **Question**:
        {question}
        """

        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 4096,
        }
        
        # Hata ayıklama için kritik loglar:
        print("INFO: Sending async request to Gemini 2.5 API...")
        
        # Run the API call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Create a wrapper function for the API call
        def generate_content_wrapper():
            return model.generate_content(prompt, generation_config=generation_config)
        
        response = await loop.run_in_executor(None, generate_content_wrapper)
        
        print("INFO: Received async response from Gemini 2.5 API.")

        # Bazen cevap döner ama içi boş olabilir (örn: güvenlik filtreleri)
        if not response.parts:
            print("WARNING: Gemini async response was empty. This might be due to safety filters or other restrictions.")
            return "I'm sorry, I couldn't generate a response for that. It might have been blocked by a safety filter."

        # Get the base response from LLM
        base_response = response.text.strip()
        print(f"--- [ASYNC LLM DEBUG] Base response length: {len(base_response)}")
        
        # Enhance response with relevant media (images/videos) - run in thread pool
        enhanced_response = await loop.run_in_executor(
            None,
            media_extractor.enhance_response_with_media,
            base_response,
            context,
            question
        )
        print(f"--- [ASYNC LLM DEBUG] Enhanced response length: {len(enhanced_response)}")
        print(f"--- [ASYNC LLM DEBUG] Media added: {len(enhanced_response) > len(base_response)}")
        
        return enhanced_response

    except Exception as e:
        print("\n" + "="*50)
        print("!!!!!! FATAL ERROR IN ASYNC GEMINI API CALL !!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("="*50 + "\n")
        
        return "Sorry, I encountered an error while communicating with the language model. The technical team has been notified. Please check the server logs for details."
def get_available_gemini_models_markdown():
    models = []
    for m in genai.list_models():
        if 'generateContent' in getattr(m, 'supported_generation_methods', []):
            models.append(m.name)
    if not models:
        return "No Gemini models available."
    return "### Available Gemini Models\n" + "\n".join(f"- `{name}`" for name in models) 