from flask import Blueprint, request, jsonify
from transformers import pipeline

# ------------------------
# Load model ONCE (startup)
# ------------------------
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# ------------------------
# Blueprint
# ------------------------
sentiment_bp = Blueprint("sentiment", __name__, url_prefix="/sentiment")

# ------------------------
# REST endpoint
# ------------------------
@sentiment_bp.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()

    if len(text) == 0:
        return jsonify({"error": "Empty text"}), 400

    # Run model
    result = sentiment_model(text)[0]

    label = result["label"]
    score = float(result["score"])

    # ------------------------
    # Neutral sentiment logic
    # ------------------------
    if score < 0.6:
        sentiment = "neutral"
    else:
        sentiment = label.lower()

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(score, 3),
        "raw_label": label
    })
