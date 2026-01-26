import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import PyPDF2

load_dotenv()

app = Flask(__name__)
CORS(app)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

@app.route("/")
def home():
    return "EDU AI Backend Running"

# ---------------- CHAT ----------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful educational AI assistant."},
            {"role": "user", "content": data["message"]}
        ]
    )

    return jsonify({"reply": response.choices[0].message.content})


# ---------------- PDF ----------------
@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["file"]

    reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Summarize this PDF in simple words"},
            {"role": "user", "content": text[:6000]}
        ]
    )

    return jsonify({
        "summary": response.choices[0].message.content
    })


if __name__ == "__main__":
    app.run()
