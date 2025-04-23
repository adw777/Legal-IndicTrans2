# GEMINI - FOR TRANSLATION (ENGLISH TO HINDI)
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from legalTransGemini import translate_english_to_hindi_legal
import uvicorn
from asgiref.wsgi import WsgiToAsgi

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    """Endpoint to translate English text to Hindi legal text"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Validate input
    if 'text' not in data or not data['text']:
        return jsonify({"error": "Text field is required"}), 400
    
    # Get API key from environment or request
    # api_key = os.getenv("GEMINI_API_KEY")
    api_key = "AIzaSyDizZsf_WytJf8qsA7F_ihTHZRLwb1Mz40"
    if not api_key:
        if 'api_key' not in data or not data['api_key']:
            return jsonify({"error": "API key not found. Provide it in the request or set GEMINI_API_KEY environment variable"}), 400
        api_key = data['api_key']
    
    try:
        # Translate the text
        hindi_translation = translate_english_to_hindi_legal(data['text'], api_key)
        return jsonify({"translation": hindi_translation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok"})

# Create ASGI app
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(asgi_app, host="0.0.0.0", port=port, log_level="info")