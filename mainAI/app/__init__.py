from flask import Flask, render_template
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    from .routes import main
    app.register_blueprint(main)

    # 👉 Thêm route này để trả về index.html
    @app.route("/")
    def index():
        return render_template("index.html")

    return app
