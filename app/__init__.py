from flask import Flask
from .app import bp as main_bp

def create_app():
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    app.register_blueprint(main_bp)
    return app
