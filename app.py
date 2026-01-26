import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.route("/", methods=["GET"])
def home():
    return "EDU AI Backend Running 🚀"

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

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    return jsonify({"message": "File uploaded successfully", "filename": filename})
