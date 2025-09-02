from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
client = InferenceClient(api_key=HF_API_TOKEN)

MODEL_ID = "facebook/blenderbot-400M-distill"

@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.get_json().get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Please enter a message."})
    if not HF_API_TOKEN:
        return jsonify({"reply": "Error: HF_API_TOKEN not configured."})

    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": user_message}]
        )
        bot_reply = response["choices"][0]["message"]["content"]
    except Exception as e:
        bot_reply = f"Error: Unable to reach AI service. ({e})"

    return jsonify({"reply": bot_reply})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    print("🚀 Starting Flask app on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
