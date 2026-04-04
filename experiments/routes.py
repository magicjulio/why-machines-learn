from flask import Blueprint, abort, render_template

from experiments.adaline import simulate_adaline
from experiments.catalog import CHAPTERS, EXPERIMENT_INDEX
from experiments.gradient_descent import simulate_gradient_descent
from experiments.iris_pca import simulate_iris_pca
from experiments.knearest import simulate_knearest
from experiments.perceptron import simulate_perceptron
from experiments.monty_hall import run_monty

experiments_bp = Blueprint("experiments", __name__)


@experiments_bp.get("/")
def home() -> str:
    return render_template("home.html", chapters=CHAPTERS)


@experiments_bp.get("/experiment/<slug>")
def experiment_page(slug: str) -> str:
    experiment = EXPERIMENT_INDEX.get(slug)
    if not experiment:
        abort(404)

    template_name = experiment.get("template", "experiment.html")
    return render_template(template_name, experiment=experiment)


@experiments_bp.post("/api/perceptron/simulate")
def perceptron_simulate():
    return simulate_perceptron()


@experiments_bp.post("/api/gradient-descent/run")
def gradient_descent_run():
    return simulate_gradient_descent()


@experiments_bp.post("/api/adaline/run")
def adaline_run():
    return simulate_adaline()


@experiments_bp.post("/api/monty-hall/simulate")
def monty_run():
    return run_monty()


@experiments_bp.post("/api/knearest/run")
def knearest_run():
    return simulate_knearest()


@experiments_bp.post("/api/iris-pca/run")
def iris_pca_run():
    return simulate_iris_pca()
