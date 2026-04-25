from flask import Flask, request, jsonify
from crm import get_response

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "message is required"}), 400

    user_message = data["message"]

    response = get_response(user_message)

    return jsonify({
        "user_message": user_message,
        "response": response
    })
@app.route("/")
def home():
    return "Chatbot API is running 🚀 Use /chat endpoint"

if __name__ == "__main__":
    app.run(debug=True)
