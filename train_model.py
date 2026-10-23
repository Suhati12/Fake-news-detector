"""
train_model.py
Complete Machine Learning Training and Evaluation Pipeline for Fake News Detection.
MCA Project Implementation:
- Dataset Loading (ISOT Fake & Real News Dataset)
- Data Preprocessing, Cleaning & Deduplication
- Train/Test Split (Stratified, No Data Leakage)
- Feature Extraction via TF-IDF Vectorizer
- Model Training: Multinomial Naive Bayes vs Linear Support Vector Classifier (LinearSVC)
- Comprehensive Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrices)
- Metric Visualization Charts (Matplotlib & Seaborn)
- Model Persistence using Joblib
"""

import os
import sys
import json
import gzip
import shutil
import time
import urllib.request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessor import clean_text


DATA_DIR = "data"
MODELS_DIR = "models"
STATIC_DIR = "static"

TRUE_CSV_PATH = os.path.join(DATA_DIR, "True.csv")
FAKE_CSV_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_GZ_PATH = os.path.join(DATA_DIR, "True.csv.gz")
FAKE_GZ_PATH = os.path.join(DATA_DIR, "Fake.csv.gz")

ISOT_URL_TRUE = "https://raw.githubusercontent.com/StevenPeutz/Disinformation-NLP/master/DATA/UVIC-ISOT/True.csv.gz"
ISOT_URL_FAKE = "https://raw.githubusercontent.com/StevenPeutz/Disinformation-NLP/master/DATA/UVIC-ISOT/Fake.csv.gz"


def ensure_dataset_exists():
    """Checks and ensures True.csv and Fake.csv are extracted and available."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Check if uncompressed True.csv exists; if not, check/download gz and extract
    if not os.path.exists(TRUE_CSV_PATH):
        if not os.path.exists(TRUE_GZ_PATH):
            print(f"[*] Downloading True.csv.gz from {ISOT_URL_TRUE}...")
            urllib.request.urlretrieve(ISOT_URL_TRUE, TRUE_GZ_PATH)
        print("[*] Extracting True.csv.gz...")
        with gzip.open(TRUE_GZ_PATH, "rb") as f_in, open(TRUE_CSV_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"[+] True.csv ready ({os.path.getsize(TRUE_CSV_PATH) / 1024 / 1024:.2f} MB)")

    # 2. Check if uncompressed Fake.csv exists; if not, check/download gz and extract
    if not os.path.exists(FAKE_CSV_PATH):
        if not os.path.exists(FAKE_GZ_PATH):
            print(f"[*] Downloading Fake.csv.gz from {ISOT_URL_FAKE}...")
            urllib.request.urlretrieve(ISOT_URL_FAKE, FAKE_GZ_PATH)
        print("[*] Extracting Fake.csv.gz...")
        with gzip.open(FAKE_GZ_PATH, "rb") as f_in, open(FAKE_CSV_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"[+] Fake.csv ready ({os.path.getsize(FAKE_CSV_PATH) / 1024 / 1024:.2f} MB)")


def load_and_preprocess_data(sample_size=None):
    """
    Loads True.csv and Fake.csv, assigns labels, removes duplicates and missing values,
    cleans the combined title and article text.
    
    Labels:
        0 -> Real News
        1 -> Fake News
    """
    print("\n" + "=" * 60)
    print("STEP 1: LOADING AND PREPROCESSING DATASET")
    print("=" * 60)
    
    ensure_dataset_exists()
    
    print("[*] Reading True.csv...")
    df_true = pd.read_csv(TRUE_CSV_PATH)
    df_true["label"] = 0
    df_true["label_name"] = "Real News"

    print("[*] Reading Fake.csv...")
    df_fake = pd.read_csv(FAKE_CSV_PATH)
    df_fake["label"] = 1
    df_fake["label_name"] = "Fake News"

    print(f"    Raw Real news count: {len(df_true):,}")
    print(f"    Raw Fake news count: {len(df_fake):,}")

    # Combine datasets
    df = pd.concat([df_true, df_fake], ignore_index=True)
    total_raw = len(df)
    print(f"    Total combined records: {total_raw:,}")

    # Combine title and text into a unified 'content' column
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = df["title"] + " " + df["text"]

    # Drop duplicate articles
    df.drop_duplicates(subset=["content"], inplace=True)
    print(f"    Records after duplicate removal: {len(df):,} (dropped {total_raw - len(df):,} duplicates)")

    # If sample_size is specified, take a balanced stratified sample for quick execution
    if sample_size and sample_size < len(df):
        print(f"    Sampling {sample_size:,} records for training efficiency...")
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=sample_size // 2, random_state=42)
        ).reset_index(drop=True)

    # Clean text using preprocessor
    print("[*] Cleaning and normalizing text...")
    start_time = time.time()
    df["clean_text"] = df["content"].apply(clean_text)
    print(f"    Text cleaning completed in {time.time() - start_time:.2f} seconds.")

    # Drop any records that became empty after cleaning
    df = df[df["clean_text"].str.len() > 10].copy()
    print(f"    Final clean dataset size: {len(df):,}")
    print(f"    Class distribution:\n{df['label_name'].value_counts().to_string()}")

    return df


def train_and_evaluate(df):
    """
    Executes train/test split, TF-IDF vectorization, model training,
    evaluation, chart generation, and model serialization.
    """
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE EXTRACTION (TF-IDF) & DATA SPLIT")
    print("=" * 60)

    X = np.array(df["clean_text"].tolist(), dtype=object)
    y = np.array(df["label"].tolist(), dtype=int)

    # Stratified 80/20 train/test split to preserve class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Training samples: {len(X_train):,}")
    print(f"[*] Testing samples:  {len(X_test):,}")

    # TF-IDF Vectorizer fitted STRICTLY on X_train to prevent data leakage
    print("[*] Fitting TF-IDF Vectorizer on training set...")
    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"[+] TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,} features")

    print("\n" + "=" * 60)
    print("STEP 3: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # 1. Multinomial Naive Bayes
    print("\n--- Model 1: Multinomial Naive Bayes ---")
    nb_model = MultinomialNB(alpha=0.1)
    nb_start = time.time()
    nb_model.fit(X_train_vec, y_train)
    nb_train_time = time.time() - nb_start
    print(f"[*] Naive Bayes trained in {nb_train_time:.2f}s")

    y_pred_nb = nb_model.predict(X_test_vec)
    acc_nb = float(accuracy_score(y_test, y_pred_nb))
    prec_nb = float(precision_score(y_test, y_pred_nb))
    rec_nb = float(recall_score(y_test, y_pred_nb))
    f1_nb = float(f1_score(y_test, y_pred_nb))
    cm_nb = confusion_matrix(y_test, y_pred_nb).tolist()

    print(f"    Accuracy:  {acc_nb * 100:.2f}%")
    print(f"    Precision: {prec_nb * 100:.2f}%")
    print(f"    Recall:    {rec_nb * 100:.2f}%")
    print(f"    F1-Score:  {f1_nb * 100:.2f}%")

    # 2. Linear Support Vector Classifier (Calibrated for probability estimates)
    print("\n--- Model 2: Linear Support Vector Classifier (LinearSVC) ---")
    base_svc = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    # CalibratedClassifierCV wraps LinearSVC with Platt scaling so predict_proba is available
    svc_calibrated = CalibratedClassifierCV(estimator=base_svc, cv=3)
    svc_start = time.time()
    svc_calibrated.fit(X_train_vec, y_train)
    svc_train_time = time.time() - svc_start
    print(f"[*] Calibrated LinearSVC trained in {svc_train_time:.2f}s")

    y_pred_svc = svc_calibrated.predict(X_test_vec)
    acc_svc = float(accuracy_score(y_test, y_pred_svc))
    prec_svc = float(precision_score(y_test, y_pred_svc))
    rec_svc = float(recall_score(y_test, y_pred_svc))
    f1_svc = float(f1_score(y_test, y_pred_svc))
    cm_svc = confusion_matrix(y_test, y_pred_svc).tolist()

    print(f"    Accuracy:  {acc_svc * 100:.2f}%")
    print(f"    Precision: {prec_svc * 100:.2f}%")
    print(f"    Recall:    {rec_svc * 100:.2f}%")
    print(f"    F1-Score:  {f1_svc * 100:.2f}%")

    print("\n" + "=" * 60)
    print("STEP 4: MODEL COMPARISON & SELECTION")
    print("=" * 60)
    print(f"{'Metric':<15} | {'Multinomial NB':<15} | {'LinearSVC':<15}")
    print("-" * 49)
    print(f"{'Accuracy':<15} | {acc_nb*100:>14.2f}% | {acc_svc*100:>14.2f}%")
    print(f"{'Precision':<15} | {prec_nb*100:>14.2f}% | {prec_svc*100:>14.2f}%")
    print(f"{'Recall':<15} | {rec_nb*100:>14.2f}% | {rec_svc*100:>14.2f}%")
    print(f"{'F1-Score':<15} | {f1_nb*100:>14.2f}% | {f1_svc*100:>14.2f}%")
    print("-" * 49)

    if f1_svc >= f1_nb:
        best_model_name = "LinearSVC"
        best_model = svc_calibrated
        best_f1 = f1_svc
        best_acc = acc_svc
    else:
        best_model_name = "Multinomial Naive Bayes"
        best_model = nb_model
        best_f1 = f1_nb
        best_acc = acc_nb

    print(f"\n[+] BEST MODEL SELECTED: {best_model_name} (F1: {best_f1*100:.2f}%, Accuracy: {best_acc*100:.2f}%)")

    # Save evaluation plots
    print("\n" + "=" * 60)
    print("STEP 5: GENERATING PERFORMANCE CHARTS")
    print("=" * 60)
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    generate_plots(
        acc_nb, prec_nb, rec_nb, f1_nb, cm_nb,
        acc_svc, prec_svc, rec_svc, f1_svc, cm_svc
    )

    # Save models with Joblib
    print("\n" + "=" * 60)
    print("STEP 6: SAVING TRAINED MODELS & METADATA")
    print("=" * 60)
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(nb_model, os.path.join(MODELS_DIR, "naive_bayes_model.joblib"))
    joblib.dump(svc_calibrated, os.path.join(MODELS_DIR, "linearsvc_model.joblib"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    print("[+] Models successfully saved with Joblib in models/ directory.")

    # Save JSON metrics for web dashboard display
    metrics_data = {
        "dataset_total": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features_count": len(vectorizer.vocabulary_),
        "best_model": best_model_name,
        "models": {
            "Naive Bayes": {
                "name": "Multinomial Naive Bayes",
                "accuracy": round(acc_nb * 100, 2),
                "precision": round(prec_nb * 100, 2),
                "recall": round(rec_nb * 100, 2),
                "f1_score": round(f1_nb * 100, 2),
                "confusion_matrix": cm_nb,
                "train_time_sec": round(nb_train_time, 2)
            },
            "LinearSVC": {
                "name": "Linear Support Vector Classifier",
                "accuracy": round(acc_svc * 100, 2),
                "precision": round(prec_svc * 100, 2),
                "recall": round(rec_svc * 100, 2),
                "f1_score": round(f1_svc * 100, 2),
                "confusion_matrix": cm_svc,
                "train_time_sec": round(svc_train_time, 2)
            }
        },
        "target_names": ["Real News", "Fake News"]
    }

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[+] Metrics saved to {metrics_path}")

    # Copy metrics to static for direct frontend fetch if needed
    with open(os.path.join(STATIC_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)

    print("\nTraining and evaluation pipeline completed successfully!")
    return metrics_data


def generate_plots(acc_nb, prec_nb, rec_nb, f1_nb, cm_nb,
                   acc_svc, prec_svc, rec_svc, f1_svc, cm_svc):
    """Generates comparison bar charts and confusion matrices using Seaborn and Matplotlib."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = ["#3B82F6", "#10B981"]

    # --- Plot 1: Model Metrics Comparison Bar Chart ---
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    nb_scores = [acc_nb * 100, prec_nb * 100, rec_nb * 100, f1_nb * 100]
    svc_scores = [acc_svc * 100, prec_svc * 100, rec_svc * 100, f1_svc * 100]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    rects1 = ax.bar(x - width / 2, nb_scores, width, label="Multinomial Naive Bayes", color="#4F46E5", alpha=0.9)
    rects2 = ax.bar(x + width / 2, svc_scores, width, label="LinearSVC", color="#059669", alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Comparison: Naive Bayes vs LinearSVC", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_ylim(80, 103)
    ax.legend(frameon=True, facecolor="white", shadow=True, fontsize=10)

    # Attach value labels on bars
    for rects in [rects1, rects2]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    comparison_path = os.path.join(STATIC_DIR, "model_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    print(f"[+] Saved comparison plot: {comparison_path}")

    # --- Plot 2: Side-by-Side Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    classes = ["Real (0)", "Fake (1)"]

    sns.heatmap(
        np.array(cm_nb), annot=True, fmt=",d", cmap="Blues", cbar=False,
        xticklabels=classes, yticklabels=classes, ax=axes[0],
        annot_kws={"size": 13, "weight": "bold"}
    )
    axes[0].set_title(f"Naive Bayes Confusion Matrix\nAccuracy: {acc_nb*100:.2f}%", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Actual Label", fontsize=11, fontweight="bold")

    sns.heatmap(
        np.array(cm_svc), annot=True, fmt=",d", cmap="Greens", cbar=False,
        xticklabels=classes, yticklabels=classes, ax=axes[1],
        annot_kws={"size": 13, "weight": "bold"}
    )
    axes[1].set_title(f"LinearSVC Confusion Matrix\nAccuracy: {acc_svc*100:.2f}%", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Actual Label", fontsize=11, fontweight="bold")

    plt.suptitle("Confusion Matrix Analysis on Test Set (ISOT Dataset)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    cm_path = os.path.join(STATIC_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved confusion matrices: {cm_path}")


if __name__ == "__main__":
    # Check if a sample size argument was passed (e.g. python train_model.py --sample 10000)
    sample_size = None
    if "--sample" in sys.argv:
        idx = sys.argv.index("--sample")
        if idx + 1 < len(sys.argv):
            sample_size = int(sys.argv[idx + 1])
    
    df = load_and_preprocess_data(sample_size=sample_size)
    train_and_evaluate(df)
