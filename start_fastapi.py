#!/usr/bin/env python3
"""
FastAPI Application Launcher
Run this script to start the FastAPI version of the SEM Chatbot
"""

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting SEM Chatbot FastAPI Server...")
    print("📱 Server will be available at: http://localhost:5001")
    print("📊 API Documentation: http://localhost:5001/docs")
    print("🔧 Alternative docs: http://localhost:5001/redoc")
    print("❤️  Health check: http://localhost:5001/health")
    
    uvicorn.run(
        "app_fastapi:app",  # Import string format for reload to work
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="info",
        access_log=True
    )