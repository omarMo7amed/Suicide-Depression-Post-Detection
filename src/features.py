"""
features.py
-----------
Feature engineering for Suicide & Depression Post Detection.

Provides:
  - TF-IDF vectorisation (unigram + bigram)
  - Handcrafted linguistic features
  - Feature matrix assembly
"""

import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# ── 1. TF-IDF Vectoriser ────────────────────────────────────────────────────


def build_tfidf_vectorizer(max_features: int = 50_000,
                           ngram_range: tuple = (1, 2),
                           min_df: int = 2,
                           max_df: float = 0.95) -> TfidfVectorizer:
    """
    Return a configured (unfitted) TF-IDF vectorizer.

    Parameters
    ----------
    max_features : Maximum vocabulary size.
    ngram_range  : (1, 2) → unigrams + bigrams.
    min_df       : Ignore terms appearing in fewer than min_df documents.
    max_df       : Ignore terms appearing in more than max_df fraction of docs.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,          # Apply log(1+tf) smoothing
        strip_accents='unicode',
        analyzer='word',
    )


# ── 2. Handcrafted Features ─────────────────────────────────────────────────

# Crisis / hopelessness keywords strongly associated with suicidal ideation
CRISIS_KEYWORDS = [
    'suicide', 'suicidal', 'kill', 'die', 'death', 'dead', 'end',
    'hopeless', 'worthless', 'useless', 'burden', 'pain', 'suffer',
    'hurt', 'alone', 'lonely', 'empty', 'numb', 'darkness', 'depressed',
    'depression', 'anxiety', 'tired', 'exhausted', 'goodbye', 'farewell',
    'wrist', 'rope', 'pills', 'overdose', 'method', 'plan',
    "can't", 'cannot', 'never', 'nothing', 'nobody', 'anymore',
]


def extract_handcrafted_features(texts: pd.Series) -> np.ndarray:
    """

    Features (per text):
      0  word_count            – number of words
      1  char_count            – number of characters
      2  avg_word_length       – mean word length
      3  sentence_count        – rough sentence count (split on . ! ?)
      4  exclamation_count     – number of '!'
      5  question_count        – number of '?'
      6  capital_ratio         – fraction of uppercase letters
      7  punctuation_count     – total punctuation marks
      8  unique_word_ratio     – unique words / total words
      9  crisis_keyword_count  – hits on CRISIS_KEYWORDS list
      10 negation_count        – words like "not", "never", "no", etc.
      11 first_person_count    – "i", "me", "my", "myself", "mine"


    """
    negation_re = re.compile(
        r"\b(not|never|no|neither|nor|nobody|nothing|nowhere|"
        r"don't|won't|can't|isn't|aren't|wasn't|weren't|"
        r"doesn't|didn't|shouldn't|wouldn't|couldn't)\b",
        re.IGNORECASE
    )
    first_person_re = re.compile(
        r"\b(i|me|my|myself|mine)\b", re.IGNORECASE
    )
    punct_re = re.compile(r'[^\w\s]')

    rows = []
    for raw in texts:
        raw = str(raw) if not isinstance(raw, str) else raw
        words = raw.split()
        n_words = len(words) if words else 1   # avoid div-by-zero
        n_chars = len(raw)

        avg_word_len = (sum(len(w) for w in words) / n_words) if words else 0

        sentences = re.split(r'[.!?]+', raw)
        n_sentences = max(len([s for s in sentences if s.strip()]), 1)

        n_excl = raw.count('!')
        n_ques = raw.count('?')

        alpha_chars = [c for c in raw if c.isalpha()]
        capital_ratio = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars else 0
        )

        n_punct = len(punct_re.findall(raw))

        unique_ratio = len(set(w.lower() for w in words)) / n_words

        raw_lower = raw.lower()
        crisis_hits = sum(1 for kw in CRISIS_KEYWORDS if kw in raw_lower)

        negation_hits = len(negation_re.findall(raw))
        first_person_hits = len(first_person_re.findall(raw))

        rows.append([
            n_words, n_chars, avg_word_len, n_sentences,
            n_excl, n_ques, capital_ratio, n_punct,
            unique_ratio, crisis_hits, negation_hits, first_person_hits,
        ])

    return np.array(rows, dtype=np.float32)


HANDCRAFTED_FEATURE_NAMES = [
    'word_count', 'char_count', 'avg_word_length', 'sentence_count',
    'exclamation_count', 'question_count', 'capital_ratio', 'punctuation_count',
    'unique_word_ratio', 'crisis_keyword_count', 'negation_count', 'first_person_count',
]


# ── 3. Feature Matrix Assembly ──────────────────────────────────────────────

def build_feature_matrix(clean_texts: pd.Series,
                         raw_texts: pd.Series,
                         tfidf: TfidfVectorizer,
                         fit: bool = False):
    """
    Combine TF-IDF sparse matrix with handcrafted dense features.

    Parameters
    ----------
    clean_texts : Preprocessed text (for TF-IDF).
    raw_texts   : Original text (for handcrafted features).
    tfidf       : A TfidfVectorizer instance.
    fit         : If True, fit the vectorizer on clean_texts first.

    Returns
    -------
    Sparse matrix of shape (n_samples, tfidf_features + 12)
    """
    if fit:
        tfidf_matrix = tfidf.fit_transform(clean_texts)
    else:
        tfidf_matrix = tfidf.transform(clean_texts)

    hand_matrix = csr_matrix(extract_handcrafted_features(raw_texts))
    combined = hstack([tfidf_matrix, hand_matrix])
    return combined


# ── 4. Save / Load Vectoriser ────────────────────────────────────────────────

def save_vectorizer(tfidf: TfidfVectorizer,
                    path: str = 'models/tfidf_vectorizer.joblib') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(tfidf, path)
    print(f"Vectorizer saved → {path}")


def load_vectorizer(path: str = 'models/tfidf_vectorizer.joblib') -> TfidfVectorizer:
    return joblib.load(path)


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sample_clean = pd.Series(
        ["feel hopeless want die alone", "today great day happy"])
    sample_raw = pd.Series(["I feel hopeless and want to die. I'm alone.",
                            "Today is a great day! I'm so happy!"])

    tfidf = build_tfidf_vectorizer(max_features=1000)
    X = build_feature_matrix(sample_clean, sample_raw, tfidf, fit=True)
    print("Feature matrix shape:", X.shape)
    hf = extract_handcrafted_features(sample_raw)
    print("Handcrafted features:\n", pd.DataFrame(
        hf, columns=HANDCRAFTED_FEATURE_NAMES))


# tf - how often a term appears in the document
# idf - how unique a term is across all documents (log(N / df))
# tf-idf = tf * idf, rare terms get higher weight
# idf
