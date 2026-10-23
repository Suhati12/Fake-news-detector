# Fake News Detection System 📰

An end-to-end Natural Language Processing (NLP) and Machine Learning web application developed as an **MCA Final Year Machine Learning Project**. The system accurately classifies news articles and headlines into **"Likely Real News"** or **"Likely Fake News"** using **TF-IDF Vectorization**, **Multinomial Naive Bayes**, and **Linear Support Vector Classification (LinearSVC)**.

---

## 🌟 Key Highlights & Project Features

* **Authentic Dataset**: Trained and evaluated on the University of Victoria **ISOT Fake News Dataset** (44,898 raw articles; 39,100 deduplicated records).
* **Zero Data Leakage**: Text preprocessors and TF-IDF feature extractors are fitted strictly on the training set (`X_train`) and only transform the test set (`X_test`).
* **Calibrated Probabilistic Output**: LinearSVC is calibrated using Platt scaling (`CalibratedClassifierCV`) so the web interface displays accurate confidence percentages alongside predictions.
* **Modern Web Interface**: Responsive dark-themed UI built with Flask, HTML5, and Vanilla CSS featuring glassmorphism, dynamic badges, confidence meters, preloaded examination test samples, and viva voce defense notes.
* **Persistent Models with Joblib**: Serializes models, vectorizers, and metrics to disk for fast inference without retraining on server launch.
* **Explainable NLP**: Identifies and displays influential vocabulary keywords that contributed to the model's prediction.

---

## 📊 Experimental Results & Model Comparison

Models were evaluated on a stratified 20% hold-out test set (7,820 unseen articles):

| Evaluation Metric | Multinomial Naive Bayes ($\alpha=0.1$) | Linear Support Vector Classifier (LinearSVC) | Champion Selection |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **96.07%** | **99.04%** | 🏆 **LinearSVC (+2.97%)** |
| **Precision** | **95.22%** | **98.94%** | 🏆 **LinearSVC** |
| **Recall** | **96.26%** | **98.97%** | 🏆 **LinearSVC** |
| **F1-Score** | **95.74%** | **98.95%** | 🏆 **LinearSVC (+3.21%)** |
| **Training Time** | ~0.04 seconds | ~0.60 seconds | Naive Bayes (faster) |

> **Conclusion**: **LinearSVC** is automatically selected as the Champion Model because it finds the optimal maximum-margin hyperplane in the high-dimensional sparse TF-IDF space (25,000 features), outperforming Naive Bayes which assumes word feature independence.

Visual performance charts (`model_comparison.png` and `confusion_matrices.png`) are generated automatically in `static/` during training and rendered in the web dashboard.

---

## 🏗️ System Architecture & NLP Pipeline

```
Raw News Input (Article / Headline)
        │
        ▼
Text Cleaning & Normalization (preprocessor.py)
  ├── Lowercase conversion
  ├── Removal of Reuters/news wire prefixes (avoids source leakage)
  ├── Stripping URLs, HTML tags, emails, handles
  └── Punctuation and digit sanitization
        │
        ▼
Feature Extraction (TfidfVectorizer)
  ├── Sublinear term frequency: 1 + log(tf)
  ├── N-gram range: (1, 2) [Unigrams + Bigrams]
  └── Max features: 25,000 top vocabulary tokens
        │
        ▼
Model Inference (models/best_model.joblib)
  ├── LinearSVC Hyperplane Margin / Platt Calibrated Probability
  └── Multinomial Naive Bayes Log-Likelihood
        │
        ▼
Flask REST API / Web Controller (app.py)
  ├── Authenticity Class ("Likely Real News" / "Likely Fake News")
  ├── Confidence Score Meter (e.g. 98.9%)
  ├── Discriminative Vocabulary Tokens
  └── Compliance Warning Banner
```

---

## 📁 Project Directory Structure

```
fake_news_detector/
├── app.py                      # Flask web application & REST endpoints
├── train_model.py              # ML training, evaluation & chart generator
├── preprocessor.py             # Reusable text cleaning & normalization
├── create_notebook.py          # Script to generate Jupyter notebook
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & MCA viva guide
├── data/
│   ├── True.csv                # Real news articles (ISOT benchmark)
│   └── Fake.csv                # Fake news articles (ISOT benchmark)
├── models/
│   ├── tfidf_vectorizer.joblib # Serialized TF-IDF vectorizer
│   ├── naive_bayes_model.joblib# Serialized Multinomial Naive Bayes
│   ├── linearsvc_model.joblib  # Serialized Calibrated LinearSVC
│   ├── best_model.joblib       # Auto-selected best model
│   └── metrics.json            # Model evaluation metrics & statistics
├── static/
│   ├── style.css               # Modern dark-mode styling & animations
│   ├── model_comparison.png    # Matplotlib/Seaborn bar comparison chart
│   └── confusion_matrices.png  # Confusion matrices heatmap
├── templates/
│   └── index.html              # Interactive frontend UI & Viva guide
└── notebooks/
    └── fake_news_analysis.ipynb# Complete ML workflow Jupyter notebook
```

---

## 🚀 Step-by-Step Setup & Execution Guide (VS Code)

### 1. Open the Project in VS Code
Open VS Code, press `Ctrl + K, Ctrl + O`, and select the project directory:
```
C:\Users\91821\.gemini\antigravity-ide\scratch\fake_news_detector
```

### 2. Open Terminal in VS Code
Press `` Ctrl + ` `` to open the integrated terminal in VS Code.

### 3. (Optional) Create & Activate Virtual Environment
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Or on Windows CMD:
.\venv\Scripts\activate.bat
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Train the Machine Learning Models
Execute the complete data preprocessing, training, and evaluation pipeline:
```bash
python train_model.py
```
*(If `data/True.csv` or `data/Fake.csv` are ever missing, `train_model.py` automatically downloads and extracts them for you).*

### 6. Run the Flask Web Application
```bash
python app.py
```

### 7. Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Testing the Web Application

1. **Quick Test Samples**: Click any of the preloaded sample buttons at the top of the input card (e.g., *"Real: Budget & Congress"* or *"Fake: Conspiracy Alert"*).
2. **Model Selection**: Switch between **Best Model (LinearSVC)**, **LinearSVC**, or **Multinomial Naive Bayes** from the dropdown.
3. **Run Prediction**: Click **"Predict Authenticity"** to view:
   * Prediction badge (Red for Fake, Green for Real).
   * Exact confidence percentage.
   * Model used.
   * Top influential keywords.
   * Warning disclaimer.
4. **Model Analytics Tab**: Click the **"Model Analytics & Evaluation"** tab to view actual dataset statistics, comparison tables, and generated Seaborn confusion matrix heatmaps.
5. **MCA Viva Guide Tab**: Click the **"MCA Viva Guide"** tab to view key viva answers directly on the interface during your presentation.

---

## 🎓 MCA Practical Examination & Viva Voce Q&A

### Q1: What is the TF-IDF Vectorizer and why is it used?
**Answer**: TF-IDF converts raw textual data into numerical vectors.
* **Term Frequency (TF)** measures how often a word occurs in an article.
* **Inverse Document Frequency (IDF)** diminishes the weight of universally frequent words (e.g., *the, said, would*) and amplifies distinct discriminative terms.
* In this project, we used `sublinear_tf=True` which applies logarithmic term frequency scaling ($1 + \log(TF)$) to prevent long articles with repeated keywords from skewing predictions.

### Q2: Why did LinearSVC outperform Multinomial Naive Bayes?
**Answer**:
* **Multinomial Naive Bayes** relies on the *conditional independence assumption*—it assumes words appear independently given the class. In natural language, words have strong grammatical and semantic dependencies.
* **LinearSVC** does not assume feature independence. It finds the optimal maximum-margin hyperplane separating real and fake news classes in the 25,000-dimensional sparse feature space. Since high-dimensional text representations are typically linearly separable, LinearSVC achieved a superior **99.04% accuracy** compared to Naive Bayes (**96.07%**).

### Q3: How did you prevent Data Leakage during preprocessing?
**Answer**:
* Data leakage occurs when test set distribution or vocabulary influences training.
* We strictly split the dataset **first** using `train_test_split(..., stratify=y, test_size=0.2)`.
* We fitted the `TfidfVectorizer` **only** on the training set (`vectorizer.fit_transform(X_train)`).
* The test set was evaluated **strictly using `vectorizer.transform(X_test)`** without retraining vocabulary or IDF weights.
* Additionally, common news wire source prefixes (e.g. *Reuters* tags) were stripped during text normalization to prevent the models from trivially learning publisher headers instead of stylistic cues.

### Q4: Why is F1-Score preferred over plain Accuracy?
**Answer**:
* While our dataset is relatively balanced, in real-world scenarios fake news articles are rarer than authentic news.
* **Precision** measures how many flagged articles were truly fake (preventing censorship of real news).
* **Recall** measures how many total fake news articles were successfully captured (preventing misinformation spread).
* **F1-Score** is the harmonic mean of Precision and Recall ($2 \cdot \frac{P \cdot R}{P + R}$), providing a robust single metric that penalizes extreme false positive or false negative rates.

### Q5: Why did you use Joblib instead of standard Pickle?
**Answer**:
`joblib` is optimized for serializing Python objects containing large NumPy arrays and SciPy sparse matrices. It provides efficient disk streaming and memory mapping, reducing load latency in web applications.

---

## 📜 Compliance & Ethical Notice
This software utilizes statistical Natural Language Processing to recognize textual patterns associated with historically labeled misinformation. It is designed for educational and academic research purposes and does not independently cross-reference real-time facts or external knowledge graphs.
