from flask import Blueprint, request, jsonify

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - import guard for minimal runtime compatibility
    pipeline = None

# Blueprint for sentiment API endpoints
sentiment_bp = Blueprint("sentiment", __name__)

# Lazy-loaded model cache
sentiment_model = None
model_load_error = None
model_load_attempted = False


def get_sentiment_model():
    """Load and cache the sentiment model the first time it is needed."""
    global sentiment_model, model_load_error, model_load_attempted

    if sentiment_model is not None:
        return sentiment_model

    if model_load_attempted and pipeline is None:
        return None

    if model_load_attempted and pipeline is not None:
        # Allow a retry if dependencies were installed after a failed attempt.
        model_load_attempted = False

    model_load_attempted = True
    model_load_error = None

    if pipeline is None:
        model_load_error = "transformers is not available in this environment"
        return None

    try:
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    except Exception as exc:  # pragma: no cover - runtime/dependency/environment failure
        model_load_error = str(exc)
        return None

    return sentiment_model


@sentiment_bp.route("/", methods=["GET"])
def sentiment_health():
    """Simple endpoint to verify the sentiment agent is mounted and available."""
    model = get_sentiment_model()
    return jsonify(
        {
            "service": "sentiment",
            "status": "ready" if model else "degraded",
            "model": "distilbert-base-uncased-finetuned-sst-2-english",
            "error": model_load_error,
        }
    ), (200 if model else 503)


@sentiment_bp.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    model = get_sentiment_model()
    if model is None:
        return (
            jsonify(
                {
                    "error": "Sentiment model is unavailable",
                    "details": model_load_error,
                }
            ),
            503,
        )

    result = model(text)[0]

    label = result["label"]
    score = float(result["score"])

    sentiment = "neutral" if score < 0.6 else label.lower()

    return jsonify(
        {
            "sentiment": sentiment,
            "confidence": round(score, 3),
            "raw_label": label,
        }
    )
