# Add A New Experiment (Minimal)

1. Add metadata in `experiments/catalog.py` inside the right chapter:

```python
{
    "slug": "my-new-exp",
    "title": "Experiment X: My Topic",
    "status": "Interactive",
    "description": "Short summary.",
    "template": "my_new_exp.html",
}
```

2. Create the page template in `templates/my_new_exp.html`.

3. If the experiment needs Python backend logic:
- Add logic function(s) in `experiments/` (new file or existing one).
- Add route(s) in `experiments/routes.py`.

Example API route:

```python
@experiments_bp.post("/api/my-new-exp/run")
def my_new_exp_run():
    return run_my_new_exp()
```

Current concrete example in this project:

```python
@experiments_bp.post("/api/gradient-descent/run")
def gradient_descent_run():
    return simulate_gradient_descent()
```

4. Frontend calls your API from the template JavaScript using `fetch("/api/my-new-exp/run", ...)`.

5. Start app:

```bash
source .venv/bin/activate
python app.py
```
