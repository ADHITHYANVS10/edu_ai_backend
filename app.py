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

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

PDF_TEXT = ""  # temporary storage (later we improve)

@app.before_request
def log_request():
    logging.info(f"{request.method} {request.path}")

@app.route("/", methods=["GET"])
def home():
    return "EDU AI Backend Running 🚀"

# 🔹 CHAT ONLY
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

    return jsonify({"reply": response.choices[0].message.content})


# 🔹 PDF UPLOAD
@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    global PDF_TEXT

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["file"]
    reader = PdfReader(pdf_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    PDF_TEXT = text

    return jsonify({"message": "PDF uploaded successfully"})


# 🔹 ASK FROM PDF
@app.route("/ask-pdf", methods=["POST"])
def ask_pdf():
    if not PDF_TEXT:
        return jsonify({"error": "No PDF uploaded yet"}), 400

    data = request.get_json()
    question = data.get("question")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer using the given PDF content."},
            {"role": "user", "content": f"PDF Content:\n{PDF_TEXT}\n\nQuestion:\n{question}"}
        ]
    )

    return jsonify({"reply": response.choices[0].message.content})


if __name__ == "__main__":
    app.run()
