import os
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

load_dotenv()

# 🔹 Logging setup
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# 🔹 Log every request
@app.before_request
def log_request():
    logging.info(f"{request.method} {request.path}")

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

@app.route("/")
def home():
    return "EDU AI Backend Running"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful educational AI assistant."},
            {"role": "user", "content": user_message}
        ]
    )

    return jsonify({
        "reply": response.choices[0].message.content
    })
