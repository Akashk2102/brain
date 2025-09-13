# --- CLEAN GEMINI PRO SMART ASSISTANT BACKEND ---
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/sa/chat', methods=['POST'])
def sa_chat():
    data = request.get_json()
    user_message = data.get('message', '')
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return jsonify({'message': 'Gemini API key not set on server.'}), 500
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=' + api_key
    payload = {
        "contents": [
            {"parts": [{"text": user_message}]}
        ]
    }
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        reply = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        if not reply:
            reply = 'Sorry, I could not understand.'
        return jsonify({'message': reply})
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)