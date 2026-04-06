from __future__ import annotations

from pathlib import Path

from flask import jsonify, request
import numpy as np


STORED_LABELS = np.array([5, 8], dtype=int)
ROOT_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT_DIR / "experiments" / "mnist_5_8.npz"
IMAGE_SHAPE = (40, 40)
DEFAULT_MAX_SWEEPS = 25
MAX_GALLERY_IMAGES = 5


def load_mnist_5_8() -> tuple[np.ndarray, np.ndarray, str, Path]:
    if not CACHE_PATH.exists():
        raise RuntimeError(
            "Cache file experiments/mnist_5_8.npz is missing. "
            "Place the file there or provide uploaded_image in the request payload."
        )

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


def recall_hopfield(
    weights: np.ndarray,
    x_data: np.ndarray,
    max_sweeps: int = DEFAULT_MAX_SWEEPS,
) -> tuple[np.ndarray, list[int]]:
    state = x_data.copy()
    flips_per_sweep: list[int] = []

    for _ in range(max_sweeps):
        flips = 0
        for i in range(len(state)):
            new_val = 1 if np.dot(weights[i], state) >= 0 else -1
            if new_val != state[i]:
                state[i] = new_val
                flips += 1

        flips_per_sweep.append(flips)

        if flips == 0:
            break

    return state, flips_per_sweep


def nearest_stored_index(
    x_data: np.ndarray,
    stored_patterns: np.ndarray,
) -> tuple[int, int]:
    distances = np.sum(stored_patterns != x_data, axis=1)
    idx = int(np.argmin(distances))
    return idx, int(distances[idx])


def is_exact_stored_pattern(x_data: np.ndarray, stored_patterns: np.ndarray) -> bool:
    return bool(np.any(np.all(stored_patterns == x_data, axis=1)))


def hopfield_energy(weights: np.ndarray, state: np.ndarray) -> float:
    return float(-0.5 * state @ weights @ state)


def pattern_to_image(pattern: np.ndarray) -> list[list[int]]:
    image = pattern.reshape(IMAGE_SHAPE)
    # Convert {-1, +1} to {0, 1} for display.
    image = ((image + 1) // 2).astype(int)
    return image.tolist()


def _resize_nearest(image: np.ndarray, target_shape: tuple[int, int] = IMAGE_SHAPE) -> np.ndarray:
    src_h, src_w = image.shape
    dst_h, dst_w = target_shape

    if (src_h, src_w) == (dst_h, dst_w):
        return image

    row_idx = np.linspace(0, src_h - 1, dst_h).round().astype(int)
    col_idx = np.linspace(0, src_w - 1, dst_w).round().astype(int)
    return image[row_idx][:, col_idx]


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr

    if arr.ndim != 3:
        raise ValueError("uploaded_image must be 2D or 3D image data.")

    # H x W x C
    if arr.shape[2] in (1, 3, 4):
        if arr.shape[2] == 1:
            return arr[..., 0]
        return arr[..., :3].mean(axis=2)

    # C x H x W
    if arr.shape[0] in (1, 3, 4):
        if arr.shape[0] == 1:
            return arr[0]
        return arr[:3].mean(axis=0)

    raise ValueError("uploaded_image channel layout is unsupported.")


def _reshape_flattened(
    flat: np.ndarray,
    width: int | None,
    height: int | None,
) -> np.ndarray:
    size = flat.size

    if width is not None and height is not None and width > 0 and height > 0:
        base = width * height
        if size == base:
            return flat.reshape(height, width)
        if size == base * 3:
            return flat.reshape(height, width, 3)
        if size == base * 4:
            return flat.reshape(height, width, 4)
        raise ValueError("uploaded_image length does not match width/height metadata.")

    if size == IMAGE_SHAPE[0] * IMAGE_SHAPE[1]:
        return flat.reshape(IMAGE_SHAPE)

    side = int(np.sqrt(size))
    if side * side == size:
        return flat.reshape(side, side)

    raise ValueError("flattened uploaded_image must include width/height or be a square length.")


def parse_uploaded_image(uploaded_image: object) -> np.ndarray:
    payload = uploaded_image
    width: int | None = None
    height: int | None = None

    if isinstance(uploaded_image, dict):
        payload = uploaded_image.get("pixels", uploaded_image.get("data", uploaded_image.get("image")))
        width_val = uploaded_image.get("width")
        height_val = uploaded_image.get("height")
        width = int(width_val) if width_val is not None else None
        height = int(height_val) if height_val is not None else None

        if payload is None:
            raise ValueError("uploaded_image object must contain 'pixels', 'data', or 'image'.")

    arr = np.asarray(payload, dtype=np.float32)

    if arr.ndim == 1:
        arr = _reshape_flattened(arr, width=width, height=height)

    arr = _to_grayscale(arr)

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("uploaded_image could not be converted to a valid grayscale image.")

    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)

    if arr.min() >= 0.0 and arr.max() <= 1.0:
        arr = arr * 255.0

    arr = np.clip(arr, 0.0, 255.0)
    arr = _resize_nearest(arr, target_shape=IMAGE_SHAPE)
    return arr.astype(np.uint8)


def _parse_indices(values: object, field_name: str) -> list[int]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list of integers.")

    parsed: list[int] = []
    for idx, value in enumerate(values):
        try:
            parsed.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{idx}] must be an integer.") from exc
    return parsed


def run_hopfield():
    payload = request.get_json(silent=True) or {}

    mode = str(payload.get("mode", "preset")).strip().lower()
    if mode not in {"preset", "upload", "gallery"}:
        return {"error": "mode must be 'preset', 'upload', or 'gallery'."}, 400

    digit = int(payload.get("digit", 5))
    noise_ratio = float(payload.get("noise_ratio", 0.30))
    seed = int(payload.get("seed", 42))
    threshold = int(payload.get("threshold", 127))
    uploaded_image_payload = payload.get("uploaded_image")
    uploaded_store_payload = payload.get("uploaded_store")
    query_image_payload = payload.get("query_image")
    selected_index = payload.get("selected_index")
    included_indices_payload = payload.get("included_indices")

    if noise_ratio < 0.0 or noise_ratio > 0.55:
        return {"error": "noise_ratio must be between 0.0 and 0.55."}, 400
    if threshold < 0 or threshold > 255:
        return {"error": "threshold must be in [0, 255]."}, 400

    if mode == "gallery":
        if not isinstance(uploaded_store_payload, list):
            return {"error": "uploaded_store is required when mode='gallery' and must be a list."}, 400
        if len(uploaded_store_payload) < 1:
            return {"error": "uploaded_store must contain at least one image."}, 400
        if len(uploaded_store_payload) > MAX_GALLERY_IMAGES:
            return {"error": f"uploaded_store must contain at most {MAX_GALLERY_IMAGES} images."}, 400
        if selected_index is None:
            return {"error": "selected_index is required when mode='gallery'."}, 400

        try:
            selected_idx = int(selected_index)
        except (TypeError, ValueError):
            return {"error": "selected_index must be an integer."}, 400

        if selected_idx < 0 or selected_idx >= len(uploaded_store_payload):
            return {"error": "selected_index is out of bounds for uploaded_store."}, 400

        try:
            gallery_images = [parse_uploaded_image(image_payload) for image_payload in uploaded_store_payload]
            included_indices = _parse_indices(included_indices_payload, "included_indices")
        except ValueError as exc:
            return {"error": str(exc)}, 400

        if not included_indices:
            included_indices = list(range(len(gallery_images)))

        # Preserve order while removing duplicates.
        seen_indices: set[int] = set()
        unique_included_indices: list[int] = []
        for idx in included_indices:
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            unique_included_indices.append(idx)

        if len(unique_included_indices) < 1:
            return {"error": "included_indices must contain at least one index."}, 400

        for idx in unique_included_indices:
            if idx < 0 or idx >= len(gallery_images):
                return {"error": "included_indices contains an out-of-bounds index."}, 400

        stored_labels: list[int | str] = [f"img_{idx + 1}" for idx in unique_included_indices]
        stored_images = np.array([gallery_images[idx] for idx in unique_included_indices], dtype=np.uint8)

        if query_image_payload is not None:
            try:
                query_image = parse_uploaded_image(query_image_payload)
            except ValueError as exc:
                return {"error": str(exc)}, 400
            x_noisy = to_hopfield(query_image, threshold=threshold)
            query_source = "manual"
        else:
            x_clean_selected = to_hopfield(gallery_images[selected_idx], threshold=threshold)
            x_noisy = add_noise(x_clean_selected, noise_ratio=noise_ratio, seed=seed)
            query_source = "random"

        selected_label = f"img_{selected_idx + 1}"
        if selected_idx in unique_included_indices:
            stored_idx = unique_included_indices.index(selected_idx)
        else:
            stored_idx = 0
        tested_label: int | str = selected_label
        dataset_info = {
            "source": "gallery upload",
            "cache_path": "",
            "samples": int(len(gallery_images)),
            "memory_count": int(len(unique_included_indices)),
            "query_source": query_source,
        }
    elif mode == "upload":
        if uploaded_image_payload is None:
            return {"error": "uploaded_image is required when mode='upload'."}, 400

        try:
            uploaded_image = parse_uploaded_image(uploaded_image_payload)
        except ValueError as exc:
            return {"error": str(exc)}, 400

        stored_labels: list[int | str] = ["upload"]
        stored_images = np.array([uploaded_image], dtype=np.uint8)
        stored_idx = 0
        tested_label: int | str = "upload"
        dataset_info = {
            "source": "user upload",
            "cache_path": "",
            "samples": 1,
            "memory_count": 1,
            "query_source": "random",
        }
    else:
        if digit not in STORED_LABELS:
            return {"error": f"digit must be one of {STORED_LABELS.tolist()}."}, 400

        try:
            x_data, y_data, source, cache_path_used = load_mnist_5_8()
        except RuntimeError as exc:
            return {"error": str(exc)}, 500

        stored_labels = [int(v) for v in STORED_LABELS.tolist()]
        stored_images = np.array(
            [
                _resize_nearest(x_data[np.where(y_data == d)[0][0]], target_shape=IMAGE_SHAPE).astype(np.uint8)
                for d in STORED_LABELS
            ],
            dtype=np.uint8,
        )
        stored_idx = int(np.where(STORED_LABELS == digit)[0][0])
        tested_label = int(digit)
        dataset_info = {
            "source": source,
            "cache_path": str(cache_path_used),
            "samples": int(len(y_data)),
            "memory_count": int(len(STORED_LABELS)),
            "query_source": "random",
        }

    stored_patterns = np.array([to_hopfield(img, threshold=threshold) for img in stored_images])

    # Fresh Hopfield network for the currently learned memory set.
    weights = train_hopfield_hebbian(stored_patterns)

    x_clean = stored_patterns[stored_idx].copy()
    if mode != "gallery":
        x_noisy = add_noise(x_clean, noise_ratio=noise_ratio, seed=seed)
    x_recalled, flips_per_sweep = recall_hopfield(weights, x_noisy, max_sweeps=DEFAULT_MAX_SWEEPS)

    pred_idx, pred_distance = nearest_stored_index(x_recalled, stored_patterns)
    pred_label = stored_labels[pred_idx]
    exact_recovery = bool(np.array_equal(x_recalled, x_clean))
    is_stored_state = is_exact_stored_pattern(x_recalled, stored_patterns)

    energy_noisy = hopfield_energy(weights, x_noisy)
    energy_recalled = hopfield_energy(weights, x_recalled)

    return jsonify(
        {
            "params": {
                "mode": mode,
                "digit": digit,
                "tested_label": tested_label,
                "noise_ratio": noise_ratio,
                "max_sweeps": DEFAULT_MAX_SWEEPS,
                "seed": seed,
                "threshold": threshold,
            },
            "dataset": dataset_info,
            "stored": {
                "labels": stored_labels,
                "image_shape": list(IMAGE_SHAPE),
                "images": [pattern_to_image(p) for p in stored_patterns],
            },
            "state": {
                "clean": pattern_to_image(x_clean),
                "noisy": pattern_to_image(x_noisy),
                "recalled": pattern_to_image(x_recalled),
            },
            "history": {
                "flips_per_sweep": flips_per_sweep,
            },
            "summary": {
                "pred_label": pred_label,
                "pred_distance": pred_distance,
                "exact_recovery": exact_recovery,
                "is_stored_state": is_stored_state,
                "sweeps_executed": len(flips_per_sweep),
                "energy_noisy": float(energy_noisy),
                "energy_recalled": float(energy_recalled),
            },
        }
    )
