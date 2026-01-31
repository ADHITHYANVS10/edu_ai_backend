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
# In-memory PDF storage (per user)
# --------------------------------------------------
pdf_text_store = {}

MAX_CHARS = 3500  # safe limit for Groq free tier

# --------------------------------------------------
# Health check (IMPORTANT for Render)
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "EDU AI Backend Running 🚀", 200

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
                "You are an educational AI assistant. "
                "If the answer is found in the document, answer using it. "
                "If not found, clearly say it is not in the document and then give a general explanation."
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
    app.run(debug=True)
