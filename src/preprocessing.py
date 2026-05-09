"""
preprocessing.py
----------------
Text preprocessing pipeline for Suicide & Depression Post Detection.
Steps: lowercase → remove noise → tokenize → remove stopwords → lemmatize
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for r in resources:
        nltk.download(r, quiet=True)


# ── 1. Basic text cleaning ──────────────────────────────────────────────────

def lowercase(text: str) -> str:
    return text.lower()


def remove_noise(text: str) -> str:
    text = re.sub(r'http\S+|www\S+', '', text)          # URLs
    text = re.sub(r'<.*?>', '', text)                    # HTML tags
    # Non-alpha chars & digits
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()             # Extra whitespace
    return text


# ── 2. Tokenisation ─────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    """Word-tokenize the cleaned text using NLTK."""
    return word_tokenize(text)


# ── 3. Stopword removal ─────────────────────────────────────────────────────

STOP_WORDS = set(stopwords.words('english'))

# Keep negations — important for mental-health text
KEEP_WORDS = {
    'no', 'not', 'nor', 'never', 'neither', "don't", "won't",
    "can't", "isn't", "aren't", "wasn't", "weren't", "doesn't",
    "didn't", "shouldn't", "wouldn't", "couldn't"
}
STOP_WORDS -= KEEP_WORDS


def remove_stopwords(tokens: list) -> list:
    """Remove stopwords while preserving negation words."""
    return [t for t in tokens if t not in STOP_WORDS]


# ── 4. Lemmatisation ────────────────────────────────────────────────────────

lemmatizer = WordNetLemmatizer()


def lemmatize(tokens: list) -> list:
    return [lemmatizer.lemmatize(t) for t in tokens]


# ── 5. Full pipeline ────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:

    if not isinstance(text, str) or not text.strip():
        return ""

    text = lowercase(text)
    text = remove_noise(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame,
                         text_col: str = 'text',
                         label_col: str = 'class') -> pd.DataFrame:
    """    
    Returns
    -------
    Cleaned DataFrame with columns: ['text', 'label', 'clean_text']
    """
    df = df.copy()

    # ── Rename columns for consistency ──────────────────────────────────────
    df = df.rename(columns={text_col: 'text', label_col: 'label'})

    # ── Drop nulls & duplicates ──────────────────────────────────────────────
    df.dropna(subset=['text', 'label'], inplace=True)
    df.drop_duplicates(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Encode labels: suicide → 1, non-suicide → 0 ─────────────────────────
    df['label'] = df['label'].map({'suicide': 1, 'non-suicide': 0})

    # ── Apply text pipeline ──────────────────────────────────────────────────
    print("Applying preprocessing pipeline … (this may take a moment)")
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Drop rows where clean_text ended up empty
    df = df[df['clean_text'].str.strip() != '']
    df.reset_index(drop=True, inplace=True)

    print(f"Preprocessing complete. Rows retained: {len(df):,}")
    return df[['text', 'label', 'clean_text']]


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    download_nltk_resources()
    sample = "I can't stop feeling hopeless. Visit http://example.com for more info!!!"
    print("Raw      :", sample)
    print("Processed:", preprocess_text(sample))
