from __future__ import annotations

from flask import jsonify, request
import numpy as np


def build_linearly_separable_dataset(
    n_samples: int,
    seed: int,
    margin_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_data = rng.normal(0, 1, size=(n_samples, 2))

    true_w = np.array([1.0, -0.8])
    true_b = 0.2
    margin = x_data @ true_w + true_b
    normal = true_w / np.linalg.norm(true_w)

    mask = np.abs(margin) < margin_eps
    if np.any(mask):
        shift = (
            (margin_eps - np.abs(margin[mask]) + 1e-3)[:, None]
            * np.sign(margin[mask])[:, None]
            * normal
        )
        x_data[mask] += shift

    labels = np.where(x_data @ true_w + true_b >= 0, 1, -1)
    return x_data, labels


def train_perceptron_with_history(
    x_data: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
    max_epochs: int,
) -> tuple[np.ndarray, float, list[dict[str, object]], int]:
    w = np.zeros(x_data.shape[1], dtype=float)
    b = 0.0
    history: list[dict[str, object]] = []
    total_updates = 0

    for _ in range(max_epochs):
        mistakes = 0
        for x_i, y_i in zip(x_data, labels):
            if y_i * (float(np.dot(x_i, w)) + b) <= 0:
                w += learning_rate * y_i * x_i
                b += learning_rate * y_i
                mistakes += 1
                total_updates += 1

        history.append(
            {
                "w": w.tolist(),
                "b": float(b),
                "mistakes": int(mistakes),
            }
        )

        if mistakes == 0:
            break

    return w, b, history, total_updates


def simulate_perceptron():
    payload = request.get_json(silent=True) or {}

    n_samples = int(payload.get("n_samples", 140))
    seed = int(payload.get("seed", 0))
    margin_eps = float(payload.get("margin_eps", 0.2))
    learning_rate = float(payload.get("learning_rate", 1.0))
    max_epochs = int(payload.get("max_epochs", 40))

    if n_samples < 20 or n_samples > 800:
        return {"error": "n_samples must be between 20 and 800."}, 400
    if margin_eps < 0.01 or margin_eps > 1.0:
        return {"error": "margin_eps must be between 0.01 and 1.0."}, 400
    if learning_rate <= 0 or learning_rate > 2.0:
        return {"error": "learning_rate must be > 0 and <= 2.0."}, 400
    if max_epochs < 1 or max_epochs > 200:
        return {"error": "max_epochs must be between 1 and 200."}, 400

    x_data, labels = build_linearly_separable_dataset(
        n_samples=n_samples,
        seed=seed,
        margin_eps=margin_eps,
    )
    final_w, final_b, history, total_updates = train_perceptron_with_history(
        x_data=x_data,
        labels=labels,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
    )

    raw_scores = x_data @ final_w + final_b
    y_hat = np.where(raw_scores >= 0, 1, -1)
    accuracy = float(np.mean(y_hat == labels))

    response = {
        "params": {
            "n_samples": n_samples,
            "seed": seed,
            "margin_eps": margin_eps,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
        },
        "points": {
            "x": x_data[:, 0].tolist(),
            "y": x_data[:, 1].tolist(),
            "labels": labels.tolist(),
        },
        "history": history,
        "summary": {
            "epochs": len(history),
            "converged": bool(history and history[-1]["mistakes"] == 0),
            "total_updates": total_updates,
            "accuracy": accuracy,
            "final_w": final_w.tolist(),
            "final_b": float(final_b),
        },
    }

    return jsonify(response)
