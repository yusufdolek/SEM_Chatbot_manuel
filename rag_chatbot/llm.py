import os
import google.generativeai as genai
from .media_extractor import MediaExtractor

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
- Use a confident, professional, and helpful tone. Use relevant emojis (🚀, 📈).
- When discussing case studies, focus on objectives and results. Do NOT include Image URL or Video URL lines in your response - these will be handled automatically.
- If asked for contact info, provide: https://webtest.semtr.com/contact-us/
- If context is insufficient, state it professionally.
"""

# Initialize media extractor
media_extractor = MediaExtractor()

# Generate a friendly, concise answer using Gemini LLM
def generate_answer(question, context):
#     prompt = f"""
# # ROLE AND GOAL
# You are SEM's expert digital assistant. Your persona is professional, confident, and persuasive. Your primary goal is to provide accurate, highly readable, and well-structured answers to users based on the information provided to you.

# # CORE TASK
# Your main task is to synthesize the information from the **'PROVIDED CONTEXT'** section below to create a comprehensive answer to the user's **'QUESTION'**.
# You **MUST NOT** use any information outside of this context. Do not invent facts.

# # RESPONSE FORMATTING AND STYLE RULES
# 1.  **Clarity Over Density:** NEVER use long, dense paragraphs. Break down all complex information into smaller, digestible parts.
# 2.  **Bold Headings:** Structure your answers with short, bolded headings (e.g., `**Our Services**`) to guide the user's eye.
# 3.  **Bullet Points:** Always use bullet points (`*` or `-`) when presenting multiple items like services, benefits, facts, or client names.
# 4.  **Emphasis with Bold:** Use **bold text** (`**text**`) to highlight and emphasize SEM's key strengths, awards, partnerships (e.g., "**Google Premier Partner**"), and impressive numerical data (e.g., "**97% increase in CVR**", "**over 150 professionals**").
# 5.  **Engaging Emojis:** Use relevant emojis (like 🚀, ✨, 📈, 🎯) where appropriate to make the information more engaging and visually appealing.
# 6.  **Synthesize, Don't Copy:** Do not just copy-paste information from the context. Rephrase and synthesize it in your own words to create a natural, persuasive, and expert-sounding response.

# # BEHAVIORAL RULES
# 1.  **Language Matching:** You must respond in the same language as the user's 'QUESTION'. If the question is in Turkish, respond in Turkish. If it is in English, respond in English.
# 2.  **Handling Insufficient Information:** If the 'PROVIDED CONTEXT' does not contain the answer to the 'QUESTION', you must state it clearly and professionally. Say something like: "I couldn't find specific details on that topic in my documents, but I would be happy to help with other questions about our services."
# 3.  **Contact Information:** If the user specifically asks to speak with an expert or how to contact the company, provide this link: https://webtest.semtr.com/contact-us/

# ---
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
        # Bu log sayesinde isteğin API'ye gönderilip gönderilmediğini anlarız.
        print("INFO: Sending request to Gemini 2.5 API...")
        
        # --- API ÇAĞRISI BURADA YAPILIYOR ---
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # Bu log sayesinde API'den bir cevap alınıp alınmadığını anlarız.
        print("INFO: Received response from Gemini 2.5 API.")

        # Bazen cevap döner ama içi boş olabilir (örn: güvenlik filtreleri)
        if not response.parts:
            print("WARNING: Gemini response was empty. This might be due to safety filters or other restrictions.")
            return "I'm sorry, I couldn't generate a response for that. It might have been blocked by a safety filter."

        # Get the base response from LLM
        base_response = response.text.strip()
        print(f"--- [LLM DEBUG] Base response length: {len(base_response)}")
        
        # Enhance response with relevant media (images/videos)
        enhanced_response = media_extractor.enhance_response_with_media(
            base_response, context, question
        )
        print(f"--- [LLM DEBUG] Enhanced response length: {len(enhanced_response)}")
        print(f"--- [LLM DEBUG] Media added: {len(enhanced_response) > len(base_response)}")
        
        return enhanced_response

    except Exception as e:
        # BU BLOK EN ÖNEMLİSİ!
        # Gemini'den veya başka bir yerden gelen herhangi bir hatayı yakalar ve konsola yazdırır.
        # "Donma" sorununun nedenini burada göreceğiz.
        print("\n" + "="*50)
        print("!!!!!! FATAL ERROR IN GEMINI API CALL !!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("="*50 + "\n")
        
        # Kullanıcıya sorunun bizden kaynaklandığını belirten bir mesaj döndürür
        return "Sorry, I encountered an error while communicating with the language model. The technical team has been notified. Please check the server logs for details."
def get_available_gemini_models_markdown():
    models = []
    for m in genai.list_models():
        if 'generateContent' in getattr(m, 'supported_generation_methods', []):
            models.append(m.name)
    if not models:
        return "No Gemini models available."
    return "### Available Gemini Models\n" + "\n".join(f"- `{name}`" for name in models) 