from flask import Flask

from experiments.routes import experiments_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(experiments_bp)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
