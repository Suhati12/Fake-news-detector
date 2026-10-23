"""
app.py
Flask Web Application for Fake News Detection System.
Provides interactive UI and REST API for real-time news authenticity analysis.
MCA Final Year Project.
"""

import os
import json
import logging
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

from preprocessor import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"
STATIC_DIR = "static"
TRUE_CSV_PATH = os.path.join(DATA_DIR, "True.csv")
FAKE_CSV_PATH = os.path.join(DATA_DIR, "Fake.csv")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
NB_PATH = os.path.join(MODELS_DIR, "naive_bayes_model.joblib")
SVC_PATH = os.path.join(MODELS_DIR, "linearsvc_model.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Global variables for loaded artifacts
vectorizer = None
nb_model = None
svc_model = None
best_model = None
metrics = None
models_loaded = False


def load_artifacts():
    """Loads saved models, vectorizer, and evaluation metrics from disk."""
    global vectorizer, nb_model, svc_model, best_model, metrics, models_loaded
    try:
        if (
            os.path.exists(VECTORIZER_PATH)
            and os.path.exists(NB_PATH)
            and os.path.exists(SVC_PATH)
            and os.path.exists(BEST_MODEL_PATH)
        ):
            logger.info("Loading serialized models and vectorizer...")
            vectorizer = joblib.load(VECTORIZER_PATH)
            nb_model = joblib.load(NB_PATH)
            svc_model = joblib.load(SVC_PATH)
            best_model = joblib.load(BEST_MODEL_PATH)

            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r") as f:
                    metrics = json.load(f)
            else:
                metrics = {}

            models_loaded = True
            logger.info("All machine learning models successfully loaded into memory.")
        else:
            models_loaded = False
            logger.warning(
                "Model files not found in models/. Please run `python train_model.py` first."
            )
    except Exception as e:
        models_loaded = False
        logger.error(f"Error loading models: {str(e)}")


# Load models on application startup
load_artifacts()


def get_model_by_choice(choice: str):
    """Returns the selected model instance and its readable name."""
    if choice == "naive_bayes":
        return nb_model, "Multinomial Naive Bayes"
    elif choice == "linearsvc":
        return svc_model, "Linear Support Vector Classifier"
    else:
        best_name = metrics.get("best_model", "Best Evaluated Model") if metrics else "Best Model (LinearSVC)"
        return best_model, f"{best_name} (Auto-Selected Best)"


def extract_key_tokens(cleaned_text: str, top_n: int = 6):
    """
    Extracts the highest TF-IDF weighted tokens present in the input text
    to explain the model's pattern recognition to the user.
    """
    if not vectorizer or not cleaned_text:
        return []
    try:
        words = cleaned_text.split()
        vocab = vectorizer.vocabulary_
        idf = vectorizer.idf_
        
        # Find words present in vocabulary and get their IDF weights
        token_weights = []
        seen = set()
        for w in words:
            if w in vocab and w not in seen:
                seen.add(w)
                token_weights.append((w, idf[vocab[w]]))
                
        # Sort by weight descending
        token_weights.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in token_weights[:top_n]]
    except Exception:
        return []


@app.route("/", methods=["GET"])
def index():
    """Renders the main fake news prediction interface."""
    # Attempt reload if models were previously missing
    if not models_loaded:
        load_artifacts()

    return render_template(
        "index.html",
        models_loaded=models_loaded,
        metrics=metrics or {}
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    API & Form Endpoint to predict news authenticity.
    Accepts JSON or Form data with 'news_text' and optional 'model_choice'.
    """
    # Verify models are loaded
    if not models_loaded:
        load_artifacts()
        if not models_loaded:
            return jsonify({
                "success": False,
                "error": "Models are not trained yet. Please run `python train_model.py` to train and save models."
            }), 503

    # Extract input data from JSON or Form
    if request.is_json:
        data = request.get_json() or {}
        raw_text = data.get("news_text", "")
        model_choice = data.get("model_choice", "best").lower()
    else:
        raw_text = request.form.get("news_text", "")
        model_choice = request.form.get("model_choice", "best").lower()

    # Input validation
    if not raw_text or not raw_text.strip():
        return jsonify({
            "success": False,
            "error": "Input news text cannot be empty. Please paste a headline or article."
        }), 400

    # Text Preprocessing
    cleaned = clean_text(raw_text)
    if len(cleaned.split()) < 3:
        return jsonify({
            "success": False,
            "error": "The input text is too short or contains no meaningful words after preprocessing. Please enter at least a full headline."
        }), 400

    # Select model
    model, model_name = get_model_by_choice(model_choice)
    if model is None:
        return jsonify({
            "success": False,
            "error": "The selected model is unavailable."
        }), 500

    try:
        # Transform using pre-fitted TF-IDF
        features = vectorizer.transform([cleaned])

        # Predict class (0 = Real News, 1 = Fake News)
        prediction_class = int(model.predict(features)[0])
        is_fake = bool(prediction_class == 1)

        # Calculate confidence probability
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            confidence = float(probs[prediction_class]) * 100
        elif hasattr(model, "decision_function"):
            score = model.decision_function(features)[0]
            # Sigmoid approximation for decision function margin
            prob_fake = 1 / (1 + np.exp(-score))
            confidence = float(prob_fake if is_fake else (1 - prob_fake)) * 100
        else:
            confidence = 85.0

        confidence = round(confidence, 2)
        prediction_label = "Likely Fake News" if is_fake else "Likely Real News"

        # Extract top influential tokens
        key_tokens = extract_key_tokens(cleaned)

        result = {
            "success": True,
            "prediction": prediction_label,
            "is_fake": is_fake,
            "confidence": confidence,
            "model_used": model_name,
            "clean_snippet": (cleaned[:160] + "...") if len(cleaned) > 160 else cleaned,
            "word_count": len(raw_text.split()),
            "key_tokens": key_tokens,
            "disclaimer": "Notice: This prediction is based on statistical machine learning patterns learned from historical news datasets and does not independently verify factual accuracy."
        }

        # If AJAX request, return JSON
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify(result)

        # Standard form fallback
        return render_template(
            "index.html",
            models_loaded=models_loaded,
            metrics=metrics or {},
            result=result,
            input_text=raw_text,
            selected_model=model_choice
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"An error occurred during prediction: {str(e)}"
        }), 500


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns the evaluation metrics JSON."""
    if metrics:
        return jsonify({"success": True, "metrics": metrics})
    return jsonify({"success": False, "error": "Metrics unavailable"}), 404


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "dataset_available": os.path.exists(TRUE_CSV_PATH) and os.path.exists(FAKE_CSV_PATH)
    })


if __name__ == "__main__":
    # Run server on port 5000 with threading enabled
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
