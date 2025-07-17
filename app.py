# app.py - Main FastAPI application

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
import traceback
from rag_chatbot.chatbot import get_chatbot_response

# Load environment variables from .env
load_dotenv()

# Debug: Check if API key is loaded
import os
print(f"✅ GEMINI_API_KEY loaded: {'YES' if os.getenv('GEMINI_API_KEY') else 'NO'}")

app = FastAPI(
    title="SEM Chatbot API",
    description="Smart Assistant for SEM",
    version="2.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/backpage", response_class=HTMLResponse)
async def backpage(request: Request):
    """Serve the backpage"""
    return templates.TemplateResponse("backpage.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    """
    Chat endpoint - processes user messages and returns AI responses
    """
    try:
        user_message = chat_message.message
        print(f"\n--- NEW REQUEST RECEIVED ---")
        print(f"User Message: {user_message}")
        
        # Call chatbot response function
        response_text = await get_chatbot_response(user_message)
        
        print("Step 7: Final response generated, sending to frontend.")
        return ChatResponse(response=response_text)
        
    except Exception as e:
        print(f"\n!!!!!! FATAL ERROR IN CHAT ROUTE !!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500, 
            detail="A critical error occurred on the server. Please check the logs."
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "SEM Chatbot API is running"}

if __name__ == '__main__':
    import uvicorn
    
    print("🚀 Starting SEM Chatbot Server...")
    print("📱 Server will be available at: http://localhost:5001")
    print("📊 API Documentation: http://localhost:5001/docs")
    print("🔧 Alternative docs: http://localhost:5001/redoc")
    print("❤️  Health check: http://localhost:5001/health")
    print("-" * 50)
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=5001, 
        reload=True,
        log_level="info",
        access_log=True
    )