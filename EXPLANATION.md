# Suicide & Depression Post Detection — Technical Explanation

## Overview

This system classifies Reddit posts as suicide risk or non-suicide using NLP and machine learning.

## Text Processing

1. Clean text (remove URLs, punctuation, digits, lowercase)
2. Tokenize using NLTK word_tokenize
3. Remove stopwords (except negations like "not", "never")
4. Lemmatize using WordNet
5. Extract TF-IDF features (unigrams + bigrams)

---

## Handcrafted Features

In addition to TF-IDF, the system extracts 12 numerical features:

- Word count, character count, average word length
- Sentence count, punctuation counts
- Capital letter ratio, unique word ratio
- Crisis keyword hits, negation count, first-person pronoun count

---

## Classification Models

Six ML models are trained:

1. Logistic Regression
2. SVM (LinearSVC)
3. Decision Tree
4. Random Forest
5. AdaBoost
6. K-Nearest Neighbors

Each model uses stratified 5-fold cross-validation with class balancing.

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion matrices
- Focus on high recall to minimize false negatives

---

## LLM Integration

Groq's LLaMA-3 model provides explanations for predictions, highlighting linguistic signals and confidence levels.
