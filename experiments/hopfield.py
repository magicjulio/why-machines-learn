from __future__ import annotations

from pathlib import Path

from flask import jsonify, request
import numpy as np


STORED_LABELS = np.array([5, 8], dtype=int)
ROOT_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT_DIR / "experiments" / "mnist_5_8.npz"
IMAGE_SHAPE = (28, 28)


def load_mnist_5_8() -> tuple[np.ndarray, np.ndarray, str, Path]:
    

    data = np.load(CACHE_PATH)
    x_data = np.asarray(data["X"], dtype=np.uint8)
    y_data = np.asarray(data["y"], dtype=int)
    return x_data, y_data, "local cache", CACHE_PATH



def to_hopfield(x_data: np.ndarray, threshold: int = 127) -> np.ndarray:
    return np.where(x_data > threshold, 1, -1).astype(np.int8).reshape(-1)


def train_hopfield_hebbian(patterns: np.ndarray) -> np.ndarray:
    n_neurons = patterns.shape[1]
    weights = np.zeros((n_neurons, n_neurons), dtype=np.float32)

    for pattern in patterns.astype(np.float32):
        weights += np.outer(pattern, pattern)

    weights /= float(len(patterns))
    np.fill_diagonal(weights, 0)
    return weights


def add_noise(pattern: np.ndarray, noise_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = pattern.copy()

    n_flip = int(len(pattern) * noise_ratio)
    if n_flip == 0:
        return noisy

    idx = rng.choice(len(pattern), size=n_flip, replace=False)
    noisy[idx] *= -1
    return noisy


def recall_hopfield_with_history(
    weights: np.ndarray,
    x_data: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    state = x_data.copy()
    history = [state.copy()]
    flips_per_sweep: list[int] = []

    for _ in range(steps):
        flips = 0
        for i in range(len(state)):
            new_val = 1 if np.dot(weights[i], state) >= 0 else -1
            if new_val != state[i]:
                state[i] = new_val
                flips += 1

        flips_per_sweep.append(flips)
        history.append(state.copy())

        if flips == 0:
            break

    return state, history, flips_per_sweep


def nearest_stored_label(
    x_data: np.ndarray,
    stored_patterns: np.ndarray,
    stored_labels: np.ndarray,
) -> tuple[int, int]:
    distances = np.sum(stored_patterns != x_data, axis=1)
    idx = int(np.argmin(distances))
    return int(stored_labels[idx]), int(distances[idx])


def is_exact_stored_pattern(x_data: np.ndarray, stored_patterns: np.ndarray) -> bool:
    return bool(np.any(np.all(stored_patterns == x_data, axis=1)))


def hopfield_energy(weights: np.ndarray, state: np.ndarray) -> float:
    return float(-0.5 * state @ weights @ state)


def pattern_to_image(pattern: np.ndarray) -> list[list[int]]:
    image = pattern.reshape(IMAGE_SHAPE)
    # Convert {-1, +1} to {0, 1} for display.
    image = ((image + 1) // 2).astype(int)
    return image.tolist()


def run_hopfield():
    payload = request.get_json(silent=True) or {}

    digit = int(payload.get("digit", 5))
    noise_ratio = float(payload.get("noise_ratio", 0.30))
    steps = int(payload.get("steps", 25))
    seed = int(payload.get("seed", 42))
    threshold = int(payload.get("threshold", 127))

    if digit not in STORED_LABELS:
        return {"error": f"digit must be one of {STORED_LABELS.tolist()}."}, 400
    if noise_ratio < 0.0 or noise_ratio > 0.55:
        return {"error": "noise_ratio must be between 0.0 and 0.55."}, 400
    if steps < 1 or steps > 80:
        return {"error": "steps must be between 1 and 80."}, 400
    if threshold < 0 or threshold > 255:
        return {"error": "threshold must be in [0, 255]."}, 400

    try:
        x_data, y_data, source, cache_path_used = load_mnist_5_8()
    except RuntimeError as exc:
        return {"error": str(exc)}, 500

    stored_images = np.array([x_data[np.where(y_data == d)[0][0]] for d in STORED_LABELS])
    stored_patterns = np.array([to_hopfield(img, threshold=threshold) for img in stored_images])

    weights = train_hopfield_hebbian(stored_patterns)

    stored_idx = int(np.where(STORED_LABELS == digit)[0][0])
    x_clean = stored_patterns[stored_idx].copy()
    x_noisy = add_noise(x_clean, noise_ratio=noise_ratio, seed=seed)
    x_recalled, history, flips_per_sweep = recall_hopfield_with_history(weights, x_noisy, steps=steps)

    pred_label, pred_distance = nearest_stored_label(x_recalled, stored_patterns, STORED_LABELS)
    exact_recovery = bool(np.array_equal(x_recalled, x_clean))
    is_stored_state = is_exact_stored_pattern(x_recalled, stored_patterns)

    energies = [hopfield_energy(weights, state) for state in history]
    distance_to_target = [int(np.sum(state != x_clean)) for state in history]

    return jsonify(
        {
            "params": {
                "digit": digit,
                "noise_ratio": noise_ratio,
                "steps": steps,
                "seed": seed,
                "threshold": threshold,
            },
            "dataset": {
                "source": source,
                "cache_path": str(cache_path_used),
                "samples": int(len(y_data)),
            },
            "stored": {
                "labels": STORED_LABELS.tolist(),
                "image_shape": list(IMAGE_SHAPE),
                "images": [pattern_to_image(p) for p in stored_patterns],
            },
            "state": {
                "clean": pattern_to_image(x_clean),
                "noisy": pattern_to_image(x_noisy),
                "recalled": pattern_to_image(x_recalled),
            },
            "history": {
                "steps": list(range(len(history))),
                "images": [pattern_to_image(state) for state in history],
                "energies": energies,
                "distance_to_target": distance_to_target,
                "flips_per_sweep": flips_per_sweep,
            },
            "summary": {
                "pred_label": pred_label,
                "pred_distance": pred_distance,
                "exact_recovery": exact_recovery,
                "is_stored_state": is_stored_state,
                "sweeps_executed": len(history) - 1,
                "energy_noisy": float(energies[0]),
                "energy_recalled": float(energies[-1]),
            },
        }
    )
