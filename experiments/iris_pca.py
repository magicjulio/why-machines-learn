from __future__ import annotations

import csv
from pathlib import Path

from flask import jsonify, request
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
IRIS_CSV = BASE_DIR / "iris.csv"
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_iris_dataset() -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    species: list[str] = []

    with IRIS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([float(row[name]) for name in FEATURE_NAMES])
            species.append(row["species"])

    return np.asarray(rows, dtype=float), np.asarray(species, dtype=object)


def run_pca(x_data: np.ndarray, center_data: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_vector = np.mean(x_data, axis=0)
    x_centered = x_data - mean_vector if center_data else x_data.copy()

    covariance = (x_centered.T @ x_centered) / float(len(x_centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    transformed = x_centered @ eigenvectors

    return transformed, eigenvalues, eigenvectors, mean_vector


def simulate_iris_pca():
    payload = request.get_json(silent=True) or {}

    dimensions = int(payload.get("dimensions", 2))
    center_data = bool(payload.get("center_data", True))

    if dimensions not in (2, 3):
        return {"error": "dimensions must be 2 or 3."}, 400

    x_data, species = load_iris_dataset()
    transformed, eigenvalues, eigenvectors, mean_vector = run_pca(x_data, center_data=center_data)

    explained_ratio = eigenvalues / float(np.sum(eigenvalues))
    top_projection = transformed[:, :dimensions]

    return jsonify(
        {
            "params": {
                "dimensions": dimensions,
                "center_data": center_data,
            },
            "projection": {
                "x": top_projection[:, 0].astype(float).tolist(),
                "y": top_projection[:, 1].astype(float).tolist(),
                "z": top_projection[:, 2].astype(float).tolist() if dimensions == 3 else [],
                "species": species.tolist(),
            },
            "variance": {
                "labels": [f"PC{i}" for i in range(1, len(eigenvalues) + 1)],
                "eigenvalues": eigenvalues.astype(float).tolist(),
                "explained_ratio": explained_ratio.astype(float).tolist(),
                "cumulative_ratio": np.cumsum(explained_ratio).astype(float).tolist(),
            },
            "loadings": {
                "features": FEATURE_NAMES,
                "pc1": eigenvectors[:, 0].astype(float).tolist(),
                "pc2": eigenvectors[:, 1].astype(float).tolist(),
                "pc3": eigenvectors[:, 2].astype(float).tolist(),
            },
            "summary": {
                "samples": int(len(x_data)),
                "features": int(x_data.shape[1]),
                "mean_vector": mean_vector.astype(float).tolist(),
                "pc1_ratio": float(explained_ratio[0]),
                "pc2_ratio": float(explained_ratio[1]),
                "pc3_ratio": float(explained_ratio[2]),
                "top_ratio": float(np.sum(explained_ratio[:dimensions])),
            },
        }
    )
