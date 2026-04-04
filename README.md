# The Math Behind AI - Interactive Chapters

This is a desktop-first educational site scaffold built with Flask.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## my ai process
i code the basic shape of what i want in jupyter notebooks. i give that to codex and let it implement routing and the web stuff.

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Current pages

- Homepage with chapter cards
- Chapter 1 has an interactive perceptron experiment
- Chapter 2 and Chapter 3 currently have placeholder experiment pages

## Structure

- `app.py`: app factory and blueprint registration only
- `experiments/routes.py`: experiment page and API routes
- `experiments/catalog.py`: chapter and experiment metadata
- `experiments/perceptron.py`: perceptron backend logic

## Add New Experiment with Agent

See `ADD_EXPERIMENT.md`.
