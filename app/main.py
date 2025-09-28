import os
import sys
from flask import Flask
from app.routes.churn_routes import churn_bp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")  # fallback for local dev
    app.register_blueprint(churn_bp, url_prefix="/")
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)