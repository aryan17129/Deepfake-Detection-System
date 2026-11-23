# Deepfake Detection System (mini project)

## Setup
    python -m venv venv
    venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
    pip install -r requirements.txt
    python app.py
Open http://127.0.0.1:5000

## Modes
- DEMO mode (default): no model file -> SIMULATED results, clearly labeled. Not real detection.
- MODEL mode: put a trained Keras model at model/deepfake_model.h5
  (single sigmoid output = probability of FAKE, input scaled 0-1).
  You can train a basic one with train_model.py (needs a dataset/real and dataset/fake folder).
