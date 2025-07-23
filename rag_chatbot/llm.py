import os
import google.generativeai as genai
from .media_extractor import MediaExtractor
import asyncio

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)

# Gemini model configuration
MODEL_NAME = 'gemini-2.5-flash'

SYSTEM_INSTRUCTION = """
You are SEM's expert, professional, and persuasive digital assistant. Your primary directive is to answer questions using ONLY the provided context. NEVER invent information.

Your response MUST adhere to these rules:
- Match the user's language (English or Turkish).
- Provide comprehensive, detailed, and informative answers based on the context.
- Give thorough explanations with context, background information, and relevant details.
- Format using markdown: Use **bold headings** and bullet points (`*`).
- Emphasize key data, partnerships, and achievements (e.g., "**Google Premier Partner**") in bold.
- Use a confident, professional tone with moderate emoji usage (🚀, 📈, 💡, ⭐, 🎯, 📊, 💼, 🔥, ✨, 🏆).
- When discussing case studies, provide detailed objectives, methodology, and comprehensive results with specific metrics and achievements.
- Elaborate on key points and provide context to help users understand the full picture.
- If asked for contact info, provide: https://webtest.semtr.com/contact-us/
- If user asked something relevant but the documents are not enough, state that you cannot help them and send them the contact link, professionally.
- If context is irrelevant, state that you cannot help professionally.
"""

# Media extractor for enhancing responses
media_extractor = MediaExtractor()


async def generate_answer_async(question, context):
    """Generate answer using Gemini LLM with async support"""
    try:
        # Configure Gemini 2.5 model
        model = genai.GenerativeModel(MODEL_NAME,
                                      system_instruction=SYSTEM_INSTRUCTION
                                      )

        prompt = f"""
        **Context**:
        {context}

        ---
        **Question**:
        {question}
        
        **Instructions**:
        Provide a comprehensive and detailed response to the question. Include relevant background information, context, and elaborate on key points with specific examples and metrics where available. Give thorough explanations that help the user understand the complete picture. Use the full context provided to deliver a rich, informative answer.
        """

        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 2048,  # Increased for more comprehensive responses
        }
        
        # Run the API call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def generate_content_wrapper():
            return model.generate_content(prompt, generation_config=generation_config)
        
        response = await loop.run_in_executor(None, generate_content_wrapper)

        # Response might be empty due to safety filters
        if not response.parts:
            return "I'm sorry, I couldn't generate a response for that. It might have been blocked by a safety filter."

        # Get the base response from LLM
        base_response = response.text.strip()
        
        # Enhance response with relevant media (images/videos)
        enhanced_response = await loop.run_in_executor(
            None,
            media_extractor.enhance_response_with_media,
            base_response,
            context,
            question
        )
        
        return enhanced_response

    except Exception as e:
        print("\n" + "="*50)
        print("!!!!!! FATAL ERROR IN GEMINI API CALL !!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("="*50 + "\n")
        
        return "Sorry, I encountered an error while communicating with the language model. The technical team has been notified. Please check the server logs for details."
 