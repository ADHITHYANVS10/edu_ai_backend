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
# Alfred Identity (EDVANTAIRE VERSION)
# --------------------------------------------------
SYSTEM_PROMPT = (
    "You are Alfred, the AI assistant of the Edvantaire app. "
    "If the user asks who you are, what your name is, or about your identity, "
    "you must respond exactly: "
    "'I am Alfred, the AI assistant of the Edvantaire app.' "
    "For all other questions, answer directly and concisely without introducing yourself."
)

# --------------------------------------------------
# In-memory storage
# --------------------------------------------------
pdf_text_store = {}
chat_history_store = {}

MAX_CHARS = 3500  # Safe for Groq free tier

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "Edvantaire Backend Running 🚀", 200


@app.route("/test", methods=["GET"])
def test():
    return "Alfred is online ✅", 200


# --------------------------------------------------
# Normal Chat (With Memory)
# --------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        user_message = data["message"]
        user_id = data.get("user_id")

        # Create new session if none
        if not user_id:
            user_id = str(uuid.uuid4())
            chat_history_store[user_id] = []

        # Ensure session exists
        if user_id not in chat_history_store:
            chat_history_store[user_id] = []

        # Save user message
        chat_history_store[user_id].append({
            "role": "user",
            "content": user_message
        })

        # Keep last 10 messages only
        chat_history_store[user_id] = chat_history_store[user_id][-10:]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *chat_history_store[user_id]
            ]
        )

        assistant_reply = response.choices[0].message.content

        # Save assistant reply
        chat_history_store[user_id].append({
            "role": "assistant",
            "content": assistant_reply
        })

        return jsonify({
            "reply": assistant_reply,
            "user_id": user_id
        })

    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        return jsonify({"error": "Chat failed"}), 500


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
            "message": "PDF uploaded successfully",
            "pages": len(reader.pages),
            "user_id": user_id
        })

    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return jsonify({"error": "PDF processing failed"}), 500


# --------------------------------------------------
# Ask PDF
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
                        "If the answer is not in the document, clearly say so first."
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
# Run
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)