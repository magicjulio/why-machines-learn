from __future__ import annotations

from random import choice, randint

from flask import jsonify, request


def _simulate_once(switch: bool) -> bool:
    prize_door = randint(1, 3)
    initial_choice = 1

    if prize_door == 1:
        revealed = choice([2, 3])
    else:
        revealed = 2 if prize_door == 3 else 3

    if switch:
        final_choice = 2 if revealed == 3 else 3
    else:
        final_choice = initial_choice

    return final_choice == prize_door


def run_monty():
    payload = request.get_json(silent=True) or {}

    n = int(payload.get("n", 1000))
    switch = bool(payload.get("switch", True))

    if n < 1 or n > 1_000_000:
        return {"error": "n must be between 1 and 1000000."}, 400

    wins = sum(1 for _ in range(n) if _simulate_once(switch))
    losses = n - wins

    return jsonify(
        {
            "params": {"n": n, "switch": switch},
            "result": {
                "wins": wins,
                "losses": losses,
                "win_rate": wins / float(n),
            },
        }
    )
