import os
import sys
from flask import Flask
from app.routes.churn_routes import churn_bp
from app.routes.main_routes import main_bp
from app.routes.sentiment_routes import sentiment_ui_bp
from app.agents.sentiment_agent import sentiment_bp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")  # fallback for local dev
    app.register_blueprint(main_bp, url_prefix="/")
    app.register_blueprint(churn_bp, url_prefix="/churn")
    app.register_blueprint(sentiment_bp, url_prefix="/sentiment")
    app.register_blueprint(sentiment_ui_bp, url_prefix="/")
    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="127.0.0.1", port=port)
