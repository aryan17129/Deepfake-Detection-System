"""
Deepfake Detection System - Flask backend.

Two modes:
  MODEL mode : a trained Keras model exists at model/deepfake_model.h5
  DEMO mode  : no model found -> SIMULATED results (clearly labeled, NOT real detection)
"""

import os
import uuid
import hashlib
import random

import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model", "deepfake_model.h5")

ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_FILE_SIZE_MB = 50
FRAMES_TO_ANALYZE = 5  # how many frames we sample from a video

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
# Flask automatically rejects larger requests (HTTP 413)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# 2. Load the model (if available)
# ---------------------------------------------------------------------------
model = None
if os.path.exists(MODEL_PATH):
    try:
        from tensorflow.keras.models import load_model  # imported only if needed
        model = load_model(MODEL_PATH)
        print("[INFO] Model loaded -> MODEL mode (real predictions)")
    except Exception as error:
        print(f"[WARNING] Could not load model: {error}")

MODE = "model" if model is not None else "demo"
if MODE == "demo":
    print("[INFO] No usable model -> DEMO mode (SIMULATED results)")


# ---------------------------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------------------------
def get_extension(filename):
    """Return the lowercase file extension without the dot."""
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def predict_fake_probability_model(pil_image):
    """
    REAL prediction using the trained model.
    Assumption: the model has ONE sigmoid output = probability the image is FAKE,
    trained on images scaled to 0-1 (pixel / 255).
    """
    height, width = model.input_shape[1:3]          # e.g. 128 x 128
    img = pil_image.convert("RGB").resize((width, height))
    array = np.array(img, dtype="float32") / 255.0  # scale pixels to 0-1
    array = np.expand_dims(array, axis=0)           # shape (1, h, w, 3)
    return float(model.predict(array, verbose=0)[0][0])


def predict_fake_probability_demo(file_bytes):
    """
    SIMULATED result for demo mode. This is NOT detection.
    It turns the file's hash into a repeatable pseudo-random number.
    """
    seed = int(hashlib.md5(file_bytes).hexdigest(), 16)
    return random.Random(seed).uniform(0.05, 0.95)


def extract_frames(video_path, count):
    """Grab `count` evenly spaced frames from a video using OpenCV."""
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames > 0:
        for index in np.linspace(0, total_frames - 1, count, dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()
            if success:
                # OpenCV uses BGR colors, PIL/model expect RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
    capture.release()
    return frames


def build_response(fake_probability, frames_analyzed):
    """Turn a probability into the JSON result sent to the browser."""
    is_fake = fake_probability >= 0.5
    label = "FAKE" if is_fake else "REAL"
    confidence = round(max(fake_probability, 1 - fake_probability) * 100, 1)

    if MODE == "model":
        if is_fake:
            explanation = ("The model found patterns that it associates with "
                           "AI-generated or manipulated media.")
        else:
            explanation = ("The model did not find patterns that it associates with "
                           "AI-generated or manipulated media.")
        explanation += " This is a statistical estimate, not proof."
    else:
        explanation = ("DEMO MODE: this result is SIMULATED and does NOT come from a "
                       "trained AI model. Add a trained model to get real predictions.")

    return {
        "label": label,
        "confidence": confidence,
        "explanation": explanation,
        "mode": MODE,
        "frames_analyzed": frames_analyzed,
    }


# ---------------------------------------------------------------------------
# 4. Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Serve the main page. We pass the mode so the page can show a banner."""
    return render_template("index.html", mode=MODE, max_mb=MAX_FILE_SIZE_MB)


@app.route("/analyze", methods=["POST"])
def analyze():
    """Receive an uploaded file, analyze it, return JSON."""
    # --- Validate the upload ---
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file was selected."}), 400

    extension = get_extension(file.filename)
    is_image = extension in ALLOWED_IMAGE_EXT
    is_video = extension in ALLOWED_VIDEO_EXT
    if not (is_image or is_video):
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT))
        return jsonify({"error": f"Unsupported file type. Allowed: {allowed}"}), 400

    # --- Save temporarily with a random name (never trust user filenames) ---
    temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{extension}")
    file.save(temp_path)

    try:
        with open(temp_path, "rb") as f:
            file_bytes = f.read()

        # ---------- IMAGE ----------
        if is_image:
            try:
                image = Image.open(temp_path)
                image.load()  # forces a full read -> detects corrupt files
            except Exception:
                return jsonify({"error": "The image file is corrupted or invalid."}), 400

            if MODE == "model":
                probability = predict_fake_probability_model(image)
            else:
                probability = predict_fake_probability_demo(file_bytes)
            return jsonify(build_response(probability, frames_analyzed=1))

        # ---------- VIDEO: sample a few frames ----------
        frames = extract_frames(temp_path, FRAMES_TO_ANALYZE)
        if not frames:
            return jsonify({"error": "Could not read any frames from this video."}), 400

        if MODE == "model":
            # Average the per-frame probabilities
            probabilities = [predict_fake_probability_model(fr) for fr in frames]
            probability = float(np.mean(probabilities))
        else:
            probability = predict_fake_probability_demo(file_bytes)
        return jsonify(build_response(probability, frames_analyzed=len(frames)))

    except Exception as error:
        print(f"[ERROR] {error}")
        return jsonify({"error": "Something went wrong while analyzing the file."}), 500

    finally:
        # Always delete the uploaded file, even if an error happened
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.errorhandler(413)
def file_too_large(_error):
    """Called automatically when the upload exceeds MAX_CONTENT_LENGTH."""
    return jsonify({"error": f"File is too large. Maximum size is {MAX_FILE_SIZE_MB} MB."}), 413


if __name__ == "__main__":
    app.run(debug=True)
