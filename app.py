# app.py

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import traceback # Hata izini sürmek için
from rag_chatbot.chatbot import get_chatbot_response

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # BU BLOK, UYGULAMADAKİ TÜM HATALARI YAKALAYACAK ANA KORUMA KALKANIDIR
    try:
        user_message = request.json.get('message')
        print("\n--- NEW REQUEST RECEIVED ---")
        print(f"User Message: {user_message}")
        
        # get_chatbot_response içindeki her adımı loglayacağız
        response_text = get_chatbot_response(user_message)
        
        print("Step 7: Final response generated, sending to frontend.")
        return jsonify({'response': response_text})
        
    except Exception as e:
        # EĞER PROGRAM "THINKING..."DE KALIYORSA, HATA BURAYA DÜŞECEKTİR
        print("\n!!!!!! A FATAL UNHANDLED ERROR OCCURRED IN THE CHAT ROUTE !!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        
        # Hatanın tam olarak hangi satırda olduğunu gösteren traceback'i yazdır
        traceback.print_exc()
        
        return jsonify({'response': 'A critical error occurred on the server. Please check the logs.'}), 500

if __name__ == '__main__':
    # debug=False yaparsak production ortamına daha yakın olur, ama True da kalabilir.
    app.run(debug=True, host='0.0.0.0', port=5001)