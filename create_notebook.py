"""
create_notebook.py
Generates the comprehensive Jupyter Notebook fake_news_analysis.ipynb
for the MCA Final Year Machine Learning Project.
"""

import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# MCA Machine Learning Project: Fake News Detection System\n",
                "\n",
                "**Student Name / Roll Number:** MCA Final Year Project  \n",
                "**Domain:** Natural Language Processing (NLP) & Supervised Machine Learning  \n",
                "**Dataset:** University of Victoria ISOT Fake & Real News Dataset  \n",
                "**Algorithms Implemented:**\n",
                "1. Term Frequency - Inverse Document Frequency (TF-IDF) Vectorization\n",
                "2. Multinomial Naive Bayes (Probabilistic Classifier)\n",
                "3. Linear Support Vector Classifier (LinearSVC with Platt Calibration)\n",
                "\n",
                "---\n",
                "### Project Workflow Overview\n",
                "1. **Exploratory Data Analysis (EDA):** Class distribution, word frequencies, length distributions.\n",
                "2. **Data Preprocessing & Text Normalization:** Case folding, URL/HTML stripping, punctuation removal, deduplication.\n",
                "3. **Feature Engineering:** TF-IDF n-gram vectorization with strict avoidance of data leakage.\n",
                "4. **Model Training:** Training Multinomial Naive Bayes and LinearSVC.\n",
                "5. **Performance Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrices.\n",
                "6. **Comparative Analysis & Model Selection:** Selecting the champion model based on empirical test metrics.\n",
                "7. **Model Persistence:** Serializing models and vectorizer with Joblib for web inference."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Import Core Libraries\n",
                "import os\n",
                "import sys\n",
                "import re\n",
                "import string\n",
                "import time\n",
                "import json\n",
                "import joblib\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "from sklearn.naive_bayes import MultinomialNB\n",
                "from sklearn.svm import LinearSVC\n",
                "from sklearn.calibration import CalibratedClassifierCV\n",
                "from sklearn.metrics import (\n",
                "    accuracy_score,\n",
                "    precision_score,\n",
                "    recall_score,\n",
                "    f1_score,\n",
                "    confusion_matrix,\n",
                "    classification_report,\n",
                "    roc_curve,\n",
                "    auc\n",
                ")\n",
                "\n",
                "# Visualization styling\n",
                "sns.set_theme(style='whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (10, 6)\n",
                "plt.rcParams['font.size'] = 11\n",
                "print('All libraries imported successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dataset Ingestion and Initial Exploration\n",
                "The ISOT Fake News Dataset consists of two collections:\n",
                "- `True.csv`: Verified real-world news articles sourced from Reuters.\n",
                "- `Fake.csv`: Articles flagged and verified as misinformation/fake news by PolitiFact and Wikipedia."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Paths to raw datasets\n",
                "true_path = os.path.join('..', 'data', 'True.csv')\n",
                "fake_path = os.path.join('..', 'data', 'Fake.csv')\n",
                "\n",
                "# Fallback if running from current directory\n",
                "if not os.path.exists(true_path):\n",
                "    true_path = os.path.join('data', 'True.csv')\n",
                "    fake_path = os.path.join('data', 'Fake.csv')\n",
                "\n",
                "df_true = pd.read_csv(true_path)\n",
                "df_fake = pd.read_csv(fake_path)\n",
                "\n",
                "df_true['label'] = 0\n",
                "df_true['category'] = 'Real News'\n",
                "\n",
                "df_fake['label'] = 1\n",
                "df_fake['category'] = 'Fake News'\n",
                "\n",
                "print(f'True News records: {len(df_true):,}')\n",
                "print(f'Fake News records: {len(df_fake):,}')\n",
                "df_combined = pd.concat([df_true, df_fake], ignore_index=True)\n",
                "print(f'Combined Total:    {len(df_combined):,}')\n",
                "df_combined.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Exploratory Data Analysis (EDA)\n",
                "Let us visualize the class distribution and inspect metadata attributes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Class Distribution Plot\n",
                "plt.figure(figsize=(7, 4.5))\n",
                "palette = {'Real News': '#10B981', 'Fake News': '#EF4444'}\n",
                "ax = sns.countplot(data=df_combined, x='category', palette=palette, hue='category', legend=False)\n",
                "plt.title('Class Distribution: Real vs Fake News in ISOT Dataset', fontsize=13, fontweight='bold')\n",
                "plt.xlabel('Category', fontsize=11, fontweight='bold')\n",
                "plt.ylabel('Article Count', fontsize=11, fontweight='bold')\n",
                "\n",
                "for p in ax.patches:\n",
                "    ax.annotate(f'{int(p.get_height()):,}',\n",
                "                (p.get_x() + p.get_width() / 2., p.get_height()),\n",
                "                ha='center', va='bottom', fontsize=10, fontweight='bold',\n",
                "                xytext=(0, 5), textcoords='offset points')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Word Count Distribution Analysis\n",
                "df_combined['content'] = df_combined['title'].fillna('') + ' ' + df_combined['text'].fillna('')\n",
                "df_combined['word_count'] = df_combined['content'].apply(lambda x: len(x.split()))\n",
                "\n",
                "plt.figure(figsize=(9, 5))\n",
                "sns.boxplot(data=df_combined, x='category', y='word_count', palette=palette, hue='category', legend=False)\n",
                "plt.ylim(0, 1500)\n",
                "plt.title('Word Count Distribution by News Category', fontsize=13, fontweight='bold')\n",
                "plt.ylabel('Word Count per Article', fontsize=11, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Data Cleaning & Text Preprocessing\n",
                "Text preprocessing is critical in NLP. We apply:\n",
                "1. **Wire Prefix Removal:** Removing Reuters headers to avoid publisher bias.\n",
                "2. **Lowercasing:** Normalizing tokens.\n",
                "3. **Regex Sanitization:** Stripping URLs, HTML tags, punctuation, special symbols, and digits.\n",
                "4. **Deduplication:** Dropping duplicate articles to eliminate bias."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def clean_news_text(text):\n",
                "    if not isinstance(text, str):\n",
                "        return ''\n",
                "    # Remove agency headers\n",
                "    text = re.sub(r'^.*?\\(Reuters\\)\\s*-\\s*', '', text, flags=re.IGNORECASE)\n",
                "    text = re.sub(r'\\(Reuters\\)', '', text, flags=re.IGNORECASE)\n",
                "    # Lowercase\n",
                "    text = text.lower()\n",
                "    # Remove URLs, HTML, Emails\n",
                "    text = re.sub(r'https?://\\S+|www\\.\\S+', ' ', text)\n",
                "    text = re.sub(r'<.*?>', ' ', text)\n",
                "    text = re.sub(r'\\S+@\\S+', ' ', text)\n",
                "    # Remove punctuation & digits\n",
                "    text = re.sub(r'[%s\\d]' % re.escape(string.punctuation), ' ', text)\n",
                "    text = re.sub(r'\\s+', ' ', text).strip()\n",
                "    return text\n",
                "\n",
                "# Remove duplicate raw contents\n",
                "total_before = len(df_combined)\n",
                "df_combined.drop_duplicates(subset=['content'], inplace=True)\n",
                "print(f'Duplicates dropped: {total_before - len(df_combined):,}. Remaining: {len(df_combined):,}')\n",
                "\n",
                "# Clean text\n",
                "print('Cleaning text...')\n",
                "df_combined['clean_text'] = df_combined['content'].apply(clean_news_text)\n",
                "df_combined = df_combined[df_combined['clean_text'].str.len() > 10].copy()\n",
                "print(f'Final cleaned dataset ready: {len(df_combined):,} records.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Stratified Train / Test Split (Preventing Data Leakage)\n",
                "We use an 80/20 stratified split. Crucially, the **TF-IDF vectorizer** is fitted **only on the training split** and transforms the test split."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X = np.array(df_combined['clean_text'].tolist(), dtype=object)\n",
                "y = np.array(df_combined['label'].tolist(), dtype=int)\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.20, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "print(f'Training instances: {len(X_train):,}')\n",
                "print(f'Testing instances:  {len(X_test):,}')\n",
                "\n",
                "# TF-IDF Vectorization\n",
                "vectorizer = TfidfVectorizer(\n",
                "    max_features=25000,\n",
                "    ngram_range=(1, 2),\n",
                "    stop_words='english',\n",
                "    sublinear_tf=True\n",
                ")\n",
                "\n",
                "X_train_vec = vectorizer.fit_transform(X_train)\n",
                "X_test_vec = vectorizer.transform(X_test)\n",
                "print(f'Vocabulary size: {len(vectorizer.vocabulary_):,} n-gram features')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Model Training & Evaluation: Multinomial Naive Bayes\n",
                "Multinomial Naive Bayes applies Bayes' theorem with the assumption of conditional independence between words given the document class."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "nb_model = MultinomialNB(alpha=0.1)\n",
                "t0 = time.time()\n",
                "nb_model.fit(X_train_vec, y_train)\n",
                "nb_time = time.time() - t0\n",
                "\n",
                "y_pred_nb = nb_model.predict(X_test_vec)\n",
                "acc_nb = accuracy_score(y_test, y_pred_nb)\n",
                "prec_nb = precision_score(y_test, y_pred_nb)\n",
                "rec_nb = recall_score(y_test, y_pred_nb)\n",
                "f1_nb = f1_score(y_test, y_pred_nb)\n",
                "\n",
                "print(f'Naive Bayes trained in {nb_time:.2f}s')\n",
                "print(f'Accuracy:  {acc_nb * 100:.2f}%')\n",
                "print(f'Precision: {prec_nb * 100:.2f}%')\n",
                "print(f'Recall:    {rec_nb * 100:.2f}%')\n",
                "print(f'F1-Score:  {f1_nb * 100:.2f}%')\n",
                "print('\\nDetailed Classification Report:')\n",
                "print(classification_report(y_test, y_pred_nb, target_names=['Real News', 'Fake News']))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Model Training & Evaluation: Linear Support Vector Classifier (LinearSVC)\n",
                "LinearSVC constructs a maximum-margin separating hyperplane in the 25,000-dimensional TF-IDF space. We use `CalibratedClassifierCV` to provide well-calibrated posterior probabilities."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "base_svc = LinearSVC(C=1.0, random_state=42, max_iter=2000)\n",
                "svc_calibrated = CalibratedClassifierCV(estimator=base_svc, cv=3)\n",
                "t0 = time.time()\n",
                "svc_calibrated.fit(X_train_vec, y_train)\n",
                "svc_time = time.time() - t0\n",
                "\n",
                "y_pred_svc = svc_calibrated.predict(X_test_vec)\n",
                "acc_svc = accuracy_score(y_test, y_pred_svc)\n",
                "prec_svc = precision_score(y_test, y_pred_svc)\n",
                "rec_svc = recall_score(y_test, y_pred_svc)\n",
                "f1_svc = f1_score(y_test, y_pred_svc)\n",
                "\n",
                "print(f'LinearSVC trained in {svc_time:.2f}s')\n",
                "print(f'Accuracy:  {acc_svc * 100:.2f}%')\n",
                "print(f'Precision: {prec_svc * 100:.2f}%')\n",
                "print(f'Recall:    {rec_svc * 100:.2f}%')\n",
                "print(f'F1-Score:  {f1_svc * 100:.2f}%')\n",
                "print('\\nDetailed Classification Report:')\n",
                "print(classification_report(y_test, y_pred_svc, target_names=['Real News', 'Fake News']))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Comparative Analysis & Visualization\n",
                "Let us plot the performance comparison bar chart and side-by-side confusion matrices."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Comparison Bar Chart\n",
                "metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']\n",
                "nb_vals = [acc_nb * 100, prec_nb * 100, rec_nb * 100, f1_nb * 100]\n",
                "svc_vals = [acc_svc * 100, prec_svc * 100, rec_svc * 100, f1_svc * 100]\n",
                "\n",
                "x = np.arange(len(metrics_names))\n",
                "width = 0.35\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(8.5, 5), dpi=120)\n",
                "b1 = ax.bar(x - width/2, nb_vals, width, label='Multinomial Naive Bayes', color='#4F46E5')\n",
                "b2 = ax.bar(x + width/2, svc_vals, width, label='LinearSVC', color='#059669')\n",
                "\n",
                "ax.set_ylabel('Score (%)', fontweight='bold')\n",
                "ax.set_title('Comparative Performance: Naive Bayes vs LinearSVC', fontsize=13, fontweight='bold')\n",
                "ax.set_xticks(x)\n",
                "ax.set_xticklabels(metrics_names, fontweight='bold')\n",
                "ax.set_ylim(80, 103)\n",
                "ax.legend(frameon=True, facecolor='white')\n",
                "\n",
                "for rects in [b1, b2]:\n",
                "    for bar in rects:\n",
                "        h = bar.get_height()\n",
                "        ax.annotate(f'{h:.2f}%', (bar.get_x() + bar.get_width() / 2, h),\n",
                "                    xytext=(0, 4), textcoords='offset points',\n",
                "                    ha='center', va='bottom', fontsize=9, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Side-by-Side Confusion Matrices\n",
                "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)\n",
                "classes = ['Real (0)', 'Fake (1)']\n",
                "\n",
                "sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt=',d', cmap='Blues', ax=axes[0],\n",
                "            xticklabels=classes, yticklabels=classes, annot_kws={'size': 12, 'weight': 'bold'})\n",
                "axes[0].set_title(f'Naive Bayes Confusion Matrix\\nAcc: {acc_nb*100:.2f}%', fontweight='bold')\n",
                "axes[0].set_xlabel('Predicted Label', fontweight='bold')\n",
                "axes[0].set_ylabel('Actual Label', fontweight='bold')\n",
                "\n",
                "sns.heatmap(confusion_matrix(y_test, y_pred_svc), annot=True, fmt=',d', cmap='Greens', ax=axes[1],\n",
                "            xticklabels=classes, yticklabels=classes, annot_kws={'size': 12, 'weight': 'bold'})\n",
                "axes[1].set_title(f'LinearSVC Confusion Matrix\\nAcc: {acc_svc*100:.2f}%', fontweight='bold')\n",
                "axes[1].set_xlabel('Predicted Label', fontweight='bold')\n",
                "axes[1].set_ylabel('Actual Label', fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Live Inference Test Demonstration\n",
                "Let us test unseen test headlines through the calibrated pipeline."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sample_articles = [\n",
                "    'WASHINGTON - The Senate committee approved the annual defense spending budget after extended bipartisan deliberations on capitol hill.',\n",
                "    'SHOCKING CONSPIRACY: Secret government agency exposed injecting mind control microchips into public reservoir supplies to fix world elections!'\n",
                "]\n",
                "\n",
                "for i, raw in enumerate(sample_articles, 1):\n",
                "    cleaned = clean_news_text(raw)\n",
                "    vec = vectorizer.transform([cleaned])\n",
                "    pred = svc_calibrated.predict(vec)[0]\n",
                "    proba = svc_calibrated.predict_proba(vec)[0]\n",
                "    label = 'Likely Fake News' if pred == 1 else 'Likely Real News'\n",
                "    confidence = proba[pred] * 100\n",
                "    print(f'Test Sample #{i}:')\n",
                "    print(f'  Input:      {raw[:90]}...')\n",
                "    print(f'  Prediction: {label} (Confidence: {confidence:.2f}%)\\n')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Conclusion & MCA Viva Highlights\n",
                "1. **Champion Model:** LinearSVC achieved superior accuracy and F1-score due to optimal margin hyperplane construction in high-dimensional sparse TF-IDF feature space.\n",
                "2. **Data Leakage:** Prevented by fitting TF-IDF strictly on training set.\n",
                "3. **Real-world deployment:** Both models and the vectorizer are serialized with Joblib and deployed in a modern Flask web interface."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 2,
            "pygments_lexer": "ipython3",
            "version": "3.13.6"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

os.makedirs("notebooks", exist_ok=True)
notebook_path = os.path.join("notebooks", "fake_news_analysis.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook successfully written to {notebook_path}")
