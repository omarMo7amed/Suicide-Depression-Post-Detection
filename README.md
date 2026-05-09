# Suicide & Depression Post Detection

### Project #12 — NLP + ML + LLM System

> ⚠️ **Ethical Notice**: This tool is intended for **research and educational purposes only**.
> It is NOT a substitute for professional mental health assessment.
> If you or someone you know is in crisis, please contact a qualified mental health professional.

---

## 📋 Project Overview

A complete NLP + Machine Learning pipeline that classifies Reddit posts as **Suicide Risk** or **Non-Suicide**
using the [Suicide Watch dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
(232,074 posts). The system includes:

- Full text preprocessing pipeline
- TF-IDF + handcrafted feature engineering
- Six trained ML classifiers with cross-validation & hyperparameter tuning
- Groq LLaMA-3 LLM integration for human-readable explanations
- Interactive Gradio web application (Colab-ready)
- Comprehensive EDA with 10+ visualisations

---

## 📁 Project Structure

```
suicide-depression-detector/
├── data/
│   └── Suicide_Detection.csv          # Raw Kaggle dataset (download separately)
├── notebooks/
│   └── Suicide_Depression_Detection_Pipeline.ipynb   # Full workflow notebook
├── src/
│   ├── preprocessing.py               # Text cleaning pipeline
│   ├── features.py                    # TF-IDF + handcrafted features
│   ├── train_ml.py                    # Model training & CV
│   ├── evaluate.py                    # Metrics, confusion matrix, ROC
│   └── llm.py                         # Groq LLM explanation module
├── models/                            # Saved .joblib model files
├── outputs/
│   └── plots/                         # All generated visualisation images
├── app_gradio.py                      # Gradio Colab app
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## 🚀 Quick Start (Google Colab)

### 1. Clone / Upload the project

```bash
# Option A — clone from GitHub (if hosted)
!git clone https://github.com/YOUR_USERNAME/suicide-depression-detector.git
%cd suicide-depression-detector

# Option B — upload the zip and unzip
```

### 2. Install dependencies

```bash
!pip install -r requirements.txt
```

### 3. Download the dataset

```python
# Upload your kaggle.json API key first
from google.colab import files
files.upload()   # select kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d nikhileswarkomati/suicide-watch -p data/ --unzip
```

### 4. Run the notebook

Open `notebooks/Suicide_Depression_Detection_Pipeline.ipynb` and run all cells.

### 5. Set your Groq API key (for LLM explanations)

```python

```

### 6. Launch the Gradio app

```bash
!python app_gradio.py
```

A public share link will appear — click it to use the interactive demo.

---

## 📊 Dataset

| Property  | Value                            |
| --------- | -------------------------------- |
| Source    | Kaggle — Suicide Watch           |
| Size      | 232,074 Reddit posts             |
| Classes   | `suicide` / `non-suicide`        |
| Format    | CSV (2 columns: `text`, `class`) |
| Imbalance | ~50/50 balanced                  |

---

## 🔧 Preprocessing Pipeline

```
Raw Post Text
   ↓  Lowercase
   ↓  Remove URLs, HTML, punctuation, digits
   ↓  Tokenise (NLTK word_tokenize)
   ↓  Remove stopwords (preserving negations: not, never, can't …)
   ↓  Lemmatise (WordNet)
   → Clean token string
```

---

## ⚙️ Feature Engineering

### TF-IDF (primary)

- Unigrams + Bigrams (`ngram_range=(1, 2)`)
- Up to 50,000 features
- Sublinear TF smoothing (`log(1+tf)`)

### Handcrafted Features (12 features)

| Feature                | Description                         |
| ---------------------- | ----------------------------------- |
| `word_count`           | Number of words                     |
| `char_count`           | Number of characters                |
| `avg_word_length`      | Mean word length                    |
| `sentence_count`       | Approximate sentence count          |
| `exclamation_count`    | Number of `!`                       |
| `question_count`       | Number of `?`                       |
| `capital_ratio`        | Fraction of uppercase letters       |
| `punctuation_count`    | Total punctuation marks             |
| `unique_word_ratio`    | Unique/total word ratio             |
| `crisis_keyword_count` | Hits on crisis keyword list         |
| `negation_count`       | Negation words (not, never, …)      |
| `first_person_count`   | First-person pronouns (I, me, my …) |

---

## 🤖 Models Trained

| Model               | Type              | Class Balancing           |
| ------------------- | ----------------- | ------------------------- |
| Logistic Regression | Linear            | `class_weight='balanced'` |
| SVM (LinearSVC)     | Linear kernel     | `class_weight='balanced'` |
| K-Nearest Neighbors | Instance-based    | Cosine metric             |
| Decision Tree       | Tree              | `class_weight='balanced'` |
| Random Forest       | Bagging ensemble  | `class_weight='balanced'` |
| AdaBoost            | Boosting ensemble | —                         |

**Training procedure**:

- 80/20 stratified train/test split
- 5-fold Stratified Cross-Validation
- Optional RandomizedSearchCV tuning (`tune=True`)

---

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix per model
- ROC curves (all models on one plot)
- Side-by-side model comparison bar chart

---

## 💬 LLM Integration (Groq)

Uses **Groq's LLaMA-3 8B** model to explain predictions:

```python
from src.llm import explain_prediction

explanation = explain_prediction(
    text      = "I feel completely hopeless and want to end it all.",
    label     = "Suicide Risk",
    confidence= 94.3,
    api_key   = "gsk_…"
)
print(explanation)
```

The LLM highlights:

1. Linguistic signals driving the prediction
2. Contextual nuances a human reviewer should consider
3. (For suicide-risk posts) a gentle note that professional support exists

---

## 🖥️ Gradio App Features

- Free-text input field for any Reddit-style post
- Dropdown to select which ML model to use
- Confidence slider (visual indicator)
- Optional Groq explanation (toggle + API key field)
- Example posts pre-loaded
- Similar dataset posts displayed

---

## 📦 Dependencies

See `requirements.txt`. Key packages:

```
scikit-learn, nltk, groq, gradio, pandas, numpy,
matplotlib, seaborn, wordcloud, imbalanced-learn
```

---

## 🗂️ Output Files

After running the notebook, these files are generated:

```
models/
  tfidf_vectorizer.joblib
  Logistic_Regression.joblib
  SVM_LinearSVC.joblib
  K-Nearest_Neighbors.joblib
  Decision_Tree.joblib
  Random_Forest.joblib
  AdaBoost.joblib

outputs/plots/
  class_distribution.png
  text_length_analysis.png
  boxplot_wordcount.png
  wordclouds.png
  top_words.png
  ngrams.png
  handcrafted_features.png
  correlation_heatmap.png
  cm_*.png              (per-model confusion matrices)
  roc_curves.png
  model_comparison.png
```

---

## 📌 Important Notes

1. **Run the notebook first** before launching the Gradio app — the app loads pre-trained models from disk.
2. The Groq API key is **optional** — the classification works without it; only explanations require it.
3. For very large datasets, preprocessing may take several minutes — use a GPU runtime in Colab.
4. The `tune=True` flag in `train_all_models()` enables hyperparameter search but significantly increases training time.

---

_Project #12 — NLP + ML + LLM Systems | Google Colab Edition_
