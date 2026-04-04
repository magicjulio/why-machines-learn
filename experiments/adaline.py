from __future__ import annotations

import base64
import io
from pathlib import Path

from flask import jsonify, request
import numpy as np
import soundfile as sf


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR.parent / "static"
NOISY_DIR = STATIC_DIR / "5dB"
CLEAN_DIR = STATIC_DIR / "clean"
EPSILON = 1e-8
MAX_PLOT_POINTS = 1600


def _audio_path(sample_id: int, noisy: bool) -> Path:
    if sample_id < 1 or sample_id > 27:
        raise ValueError("sample_id must be between 1 and 27.")

    if noisy:
        return NOISY_DIR / f"sp{sample_id:02d}_babble_sn5.wav"
    return CLEAN_DIR / f"sp{sample_id:02d}.wav"


def _load_pair(sample_id: int) -> tuple[np.ndarray, np.ndarray, int]:
    noisy, noisy_sr = sf.read(_audio_path(sample_id, noisy=True))
    clean, clean_sr = sf.read(_audio_path(sample_id, noisy=False))

    if noisy_sr != clean_sr:
        raise ValueError("Noisy and clean clips must share the same sample rate.")

    noisy = np.asarray(noisy, dtype=float).squeeze()
    clean = np.asarray(clean, dtype=float).squeeze()

    if noisy.ndim != 1 or clean.ndim != 1:
        raise ValueError("Expected mono audio clips.")

    n_samples = min(len(noisy), len(clean))
    return noisy[:n_samples], clean[:n_samples], noisy_sr


def run_adaline_filter(
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    taps: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = min(len(noisy_signal), len(clean_signal))
    output = np.zeros(n_samples, dtype=float)
    error = np.zeros(n_samples, dtype=float)
    learning_curve = np.zeros(n_samples, dtype=float)
    weights = np.zeros(taps, dtype=float)

    for n in range(taps - 1, n_samples):
        x_vec = noisy_signal[n - taps + 1 : n + 1][::-1]
        prediction = float(np.dot(weights, x_vec))
        residual = float(clean_signal[n] - prediction)
        norm = float(np.dot(x_vec, x_vec)) + EPSILON

        weights += (learning_rate / norm) * residual * x_vec
        output[n] = prediction
        error[n] = residual
        learning_curve[n] = residual * residual

    return output, error, learning_curve


def _rolling_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values

    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _series_for_plot(signal: np.ndarray, sample_rate: int, max_points: int = MAX_PLOT_POINTS):
    if len(signal) == 0:
        return {"t": [], "y": []}

    count = min(max_points, len(signal))
    indices = np.linspace(0, len(signal) - 1, count, dtype=int)
    times = indices / float(sample_rate)

    return {
        "t": times.tolist(),
        "y": signal[indices].astype(float).tolist(),
    }


def _audio_data_uri(signal: np.ndarray, sample_rate: int) -> str:
    waveform = np.asarray(signal, dtype=np.float32)
    peak = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
    if peak > 0:
        waveform = waveform / peak * 0.98

    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _safe_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    signal_power = float(np.mean(reference**2))
    noise_power = float(np.mean((reference - estimate) ** 2))
    return 10.0 * np.log10((signal_power + EPSILON) / (noise_power + EPSILON))


def simulate_adaline():
    payload = request.get_json(silent=True) or {}

    sample_id = int(payload.get("sample_id", 1))
    taps = int(payload.get("taps", 32))
    learning_rate = float(payload.get("learning_rate", 0.1))

    if sample_id < 1 or sample_id > 27:
        return {"error": "sample_id must be between 1 and 27."}, 400
    if taps < 4 or taps > 128:
        return {"error": "taps must be between 4 and 128."}, 400
    if learning_rate <= 0 or learning_rate > 1.5:
        return {"error": "learning_rate must be > 0 and <= 1.5."}, 400

    noisy_signal, clean_signal, sample_rate = _load_pair(sample_id)
    output_signal, error_signal, learning_curve = run_adaline_filter(
        noisy_signal=noisy_signal,
        clean_signal=clean_signal,
        taps=taps,
        learning_rate=learning_rate,
    )

    valid_slice = slice(taps - 1, None)
    smoothed_curve = _rolling_average(learning_curve[valid_slice], window=max(8, taps))

    response = {
        "params": {
            "sample_id": sample_id,
            "taps": taps,
            "learning_rate": learning_rate,
            "sample_rate": sample_rate,
        },
        "signals": {
            "clean": _series_for_plot(clean_signal, sample_rate),
            "noisy": _series_for_plot(noisy_signal, sample_rate),
            "output": _series_for_plot(output_signal, sample_rate),
        },
        "learning_curve": {
            "step": list(range(taps, len(smoothed_curve) + taps)),
            "mse": smoothed_curve.astype(float).tolist(),
        },
        "summary": {
            "duration_seconds": len(clean_signal) / float(sample_rate),
            "mse_noisy": float(np.mean((clean_signal[valid_slice] - noisy_signal[valid_slice]) ** 2)),
            "mse_output": float(np.mean((clean_signal[valid_slice] - output_signal[valid_slice]) ** 2)),
            "snr_noisy_db": float(_safe_snr(clean_signal[valid_slice], noisy_signal[valid_slice])),
            "snr_output_db": float(_safe_snr(clean_signal[valid_slice], output_signal[valid_slice])),
            "mean_absolute_error": float(np.mean(np.abs(error_signal[valid_slice]))),
        },
        "audio": {
            "clean": _audio_data_uri(clean_signal, sample_rate),
            "noisy": _audio_data_uri(noisy_signal, sample_rate),
            "output": _audio_data_uri(output_signal, sample_rate),
        },
    }

    return jsonify(response)
