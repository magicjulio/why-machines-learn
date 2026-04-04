from __future__ import annotations

from flask import jsonify, request
import numpy as np


def objective(x: float, y: float) -> float:
    return x * x + y * y


def gradient(x: float, y: float) -> tuple[float, float]:
    return 2.0 * x, 2.0 * y


def run_gradient_descent(x0: float, y0: float, learning_rate: float, steps: int):
    x = float(x0)
    y = float(y0)

    path_x = [x]
    path_y = [y]
    path_z = [objective(x, y)]

    for _ in range(steps):
        gx, gy = gradient(x, y)
        x -= learning_rate * gx
        y -= learning_rate * gy
        path_x.append(x)
        path_y.append(y)
        path_z.append(objective(x, y))

    return path_x, path_y, path_z


def simulate_gradient_descent():
    payload = request.get_json(silent=True) or {}

    x0 = float(payload.get("x0", 4.0))
    y0 = float(payload.get("y0", -3.0))
    learning_rate = float(payload.get("learning_rate", 0.15))
    steps = int(payload.get("steps", 25))

    if abs(x0) > 10 or abs(y0) > 10:
        return {"error": "x0 and y0 must be in [-10, 10]."}, 400
    if learning_rate <= 0 or learning_rate > 1.0:
        return {"error": "learning_rate must be > 0 and <= 1.0."}, 400
    if steps < 1 or steps > 150:
        return {"error": "steps must be between 1 and 150."}, 400

    path_x, path_y, path_z = run_gradient_descent(
        x0=x0,
        y0=y0,
        learning_rate=learning_rate,
        steps=steps,
    )

    grid = np.linspace(-6.0, 6.0, 70)
    mesh_x, mesh_y = np.meshgrid(grid, grid)
    mesh_z = mesh_x**2 + mesh_y**2

    response = {
        "params": {
            "x0": x0,
            "y0": y0,
            "learning_rate": learning_rate,
            "steps": steps,
        },
        "path": {
            "x": path_x,
            "y": path_y,
            "z": path_z,
        },
        "surface": {
            "x": grid.tolist(),
            "y": grid.tolist(),
            "z": mesh_z.tolist(),
        },
        "summary": {
            "start_z": float(path_z[0]),
            "final_z": float(path_z[-1]),
            "improvement": float(path_z[0] - path_z[-1]),
            "final_point": [float(path_x[-1]), float(path_y[-1])],
        },
    }

    return jsonify(response)
