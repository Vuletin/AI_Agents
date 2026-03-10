from flask import Blueprint, render_template, request
from app.agents.sentiment_agent import get_sentiment_model, model_load_error

sentiment_ui_bp = Blueprint("sentiment_ui", __name__)


@sentiment_ui_bp.route("/sentiment-ui", methods=["GET", "POST"])
def sentiment_ui():
    sentiment = None
    confidence = None
    error = None
    text = ""

    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        if not text:
            error = "Please enter some text."
        else:
            model = get_sentiment_model()
            if model is None:
                error = model_load_error or "Sentiment model is unavailable."
            else:
                result = model(text)[0]
                label = result.get("label", "unknown").lower()
                score = float(result.get("score", 0.0))
                sentiment = "neutral" if score < 0.6 else label
                confidence = round(score, 3)

    return render_template(
        "sentiment.html",
        text=text,
        sentiment=sentiment,
        confidence=confidence,
        error=error,
    )
