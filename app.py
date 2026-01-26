import os
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from PyPDF2 import PdfReader

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# Store PDF text in RAM (per user)
pdf_store = {}

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

@app.before_request
def log_request():
    logging.info(f"{request.method} {request.path}")

# ---------------------------
# NORMAL CHAT (already works)
# ---------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

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


# ---------------------------
# PDF UPLOAD
# ---------------------------
@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    user_id = request.form.get("user_id", "default_user")
    pdf_file = request.files["file"]

    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        return jsonify({"error": "Could not extract text from PDF"}), 400

    pdf_store[user_id] = text

    
    return jsonify({
    "message": "PDF uploaded and processed successfully",
    "pages": len(reader.pages)
    })


# ---------------------------
# ASK QUESTIONS FROM PDF
# ---------------------------
@app.route("/ask-pdf", methods=["POST"])
def ask_pdf():
    data = request.get_json()
    user_id = data.get("user_id", "default_user")
    question = data.get("question", "")

    if user_id not in pdf_store:
        return jsonify({"error": "No PDF uploaded for this user"}), 400

    pdf_text = pdf_store[user_id][:12000]  # limit for safety

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY using the given PDF content."
            },
            {
                "role": "user",
                "content": f"PDF CONTENT:\n{pdf_text}\n\nQUESTION:\n{question}"
            }
        ]
    )

    return jsonify({
        "answer": response.choices[0].message.content
    })


@app.route("/")
def home():
    return "EDU AI Backend Running ✅"
