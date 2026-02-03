import os
import uuid
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from PyPDF2 import PdfReader

# --------------------------------------------------
# Basic setup
# --------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --------------------------------------------------
# Alfred system identity (IMPORTANT)
# --------------------------------------------------
SYSTEM_PROMPT = (
    "You are Alfred, an AI study assistant for the EDU AI app. "
    "Do NOT introduce yourself unless the user explicitly asks "
    "who you are or what your name is. "
    "For normal questions, answer directly and concisely."
)

# --------------------------------------------------
# In-memory PDF storage (per user)
# --------------------------------------------------
pdf_text_store = {}

MAX_CHARS = 3500  # Safe for Groq free tier

# --------------------------------------------------
# Health check (Render needs this)
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "EDU AI Backend Running 🚀", 200


@app.route("/test", methods=["GET"])
def test():
    return "Alfred is online ✅", 200


# --------------------------------------------------
# Normal chat (NO PDF)
# --------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    return jsonify({
        "reply": response.choices[0].message.content
    })


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------
@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            return jsonify({"error": "PDF has no extractable text"}), 400

        text = text[:MAX_CHARS]

        user_id = str(uuid.uuid4())
        pdf_text_store[user_id] = text

        logging.info(f"PDF uploaded | pages={len(reader.pages)} | user_id={user_id}")

        return jsonify({
            "message": "PDF uploaded and processed successfully",
            "pages": len(reader.pages),
            "user_id": user_id
        })

    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return jsonify({"error": "PDF processing failed"}), 500


# --------------------------------------------------
# Ask question from PDF
# --------------------------------------------------
@app.route("/ask-pdf", methods=["POST"])
def ask_pdf():
    try:
        data = request.get_json()

        user_id = data.get("user_id")
        question = data.get("question")

        if not user_id or not question:
            return jsonify({"error": "user_id and question required"}), 400

        if user_id not in pdf_text_store:
            return jsonify({"error": "Invalid user_id"}), 400

        context = pdf_text_store[user_id]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT +
                        " Answer strictly from the provided document if possible. "
                        "If the answer is not in the document, clearly say so first, "
                        "then give a general explanation."
                    )
                },
                {
                    "role": "user",
                    "content": f"Document:\n{context}\n\nQuestion:\n{question}"
                }
            ]
        )

        return jsonify({
            "answer": response.choices[0].message.content
        })

    except Exception as e:
        logging.error(f"Q&A error: {str(e)}")
        return jsonify({"error": "Failed to answer question"}), 500


# --------------------------------------------------
# Run locally
# --------------------------------------------------
if __name__ == "__main__":
    app.run()

