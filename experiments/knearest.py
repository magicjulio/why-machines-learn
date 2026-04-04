from __future__ import annotations

from flask import jsonify, request
import numpy as np


def make_moons(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    outer_t = rng.uniform(0.0, np.pi, n_outer)
    inner_t = rng.uniform(0.0, np.pi, n_inner)

    outer = np.column_stack((np.cos(outer_t), np.sin(outer_t)))
    inner = np.column_stack((1.0 - np.cos(inner_t), 1.0 - np.sin(inner_t) - 0.5))

    x_data = np.vstack((outer, inner))
    y_data = np.concatenate((np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)))

    if noise > 0:
        x_data += rng.normal(0.0, noise, size=x_data.shape)

    perm = rng.permutation(n_samples)
    return x_data[perm], y_data[perm]


def knn_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[list[int]]]:
    predicted = np.zeros(len(query_x), dtype=int)
    neighbor_indices: list[list[int]] = []

    for i, q in enumerate(query_x):
        dists = np.linalg.norm(train_x - q, axis=1)
        idx = np.argsort(dists)[:k]
        labels = train_y[idx]
        votes = np.bincount(labels, minlength=2)
        predicted[i] = int(np.argmax(votes))
        neighbor_indices.append(idx.tolist())

    return predicted, neighbor_indices


def decision_boundary(
    train_x: np.ndarray,
    train_y: np.ndarray,
    k: int,
    grid_size: int = 90,
) -> tuple[list[float], list[float], list[list[int]]]:
    x_pad = 0.8
    y_pad = 0.8

    x_min = float(np.min(train_x[:, 0]) - x_pad)
    x_max = float(np.max(train_x[:, 0]) + x_pad)
    y_min = float(np.min(train_x[:, 1]) - y_pad)
    y_max = float(np.max(train_x[:, 1]) + y_pad)

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    z_pred, _ = knn_predict(train_x, train_y, grid_points, k)
    z = z_pred.reshape(xx.shape)

    return xs.tolist(), ys.tolist(), z.astype(int).tolist()


def simulate_knearest():
    payload = request.get_json(silent=True) or {}

    n_samples = int(payload.get("n_samples", 300))
    noise = float(payload.get("noise", 0.15))
    k = int(payload.get("k", 5))
    n_query = int(payload.get("n_query", 10))
    seed = int(payload.get("seed", 42))

    if n_samples < 40 or n_samples > 1200:
        return {"error": "n_samples must be between 40 and 1200."}, 400
    if noise < 0.0 or noise > 0.6:
        return {"error": "noise must be between 0.0 and 0.6."}, 400
    if k < 1 or k > 50:
        return {"error": "k must be between 1 and 50."}, 400
    if k > n_samples:
        return {"error": "k must be less than or equal to n_samples."}, 400
    if n_query < 1 or n_query > 60:
        return {"error": "n_query must be between 1 and 60."}, 400

    train_x, train_y = make_moons(n_samples=n_samples, noise=noise, seed=seed)
    query_x, query_true_y = make_moons(n_samples=n_query, noise=noise, seed=seed + 1)

    query_pred_y, neighbor_indices = knn_predict(train_x, train_y, query_x, k)
    xs, ys, z = decision_boundary(train_x, train_y, k=k, grid_size=90)

    accuracy = float(np.mean(query_pred_y == query_true_y))

    return jsonify(
        {
            "params": {
                "n_samples": n_samples,
                "noise": noise,
                "k": k,
                "n_query": n_query,
                "seed": seed,
            },
            "train": {
                "x": train_x[:, 0].astype(float).tolist(),
                "y": train_x[:, 1].astype(float).tolist(),
                "labels": train_y.astype(int).tolist(),
            },
            "query": {
                "x": query_x[:, 0].astype(float).tolist(),
                "y": query_x[:, 1].astype(float).tolist(),
                "true_labels": query_true_y.astype(int).tolist(),
                "pred_labels": query_pred_y.astype(int).tolist(),
                "neighbor_indices": neighbor_indices,
            },
            "boundary": {
                "x": xs,
                "y": ys,
                "z": z,
            },
            "summary": {
                "query_accuracy": accuracy,
                "correct": int(np.sum(query_pred_y == query_true_y)),
                "total": int(n_query),
            },
        }
    )
