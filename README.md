# Crop Recommendation Backend

This repository contains the Flask API for the crop recommendation system.
The trained ML model is intentionally not committed here to keep the repository lightweight.

## Local run
```bash
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Model setup
Place your trained model in `model/` or set `MODEL_URL` in the environment for cloud-hosted model storage.
