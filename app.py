from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from rag_chatbot.chatbot import get_chatbot_response
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        response = get_chatbot_response(user_message)
        return jsonify({'response': response})
    except Exception as e:
        print(f"An error occurred: {e}") # Sunucu tarafında loglama için
        return jsonify({'response': 'Sorry, an unexpected error occurred. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True) 