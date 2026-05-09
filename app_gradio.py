"""
Mental Health Post Analysis System
Web interface for suicide-risk detection using BERT embeddings, ML classification, and LLM reasoning.

Usage:
    python app_gradio.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import gradio as gr

from src.llm import explain_prediction
from src.features import build_feature_matrix, load_vectorizer
from src.preprocessing import preprocess_text, download_nltk_resources

# ── Make sure src/ is importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Download NLTK data ────────────────────────────────────────────────────────
download_nltk_resources()

# ── Load artefacts ────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_PATH = "data/suicide_detection.csv"

AVAILABLE_MODELS = {
    "Logistic Regression": "Logistic_Regression.joblib",
    "SVM (LinearSVC)":     "SVM_LinearSVC.joblib",
    "Random Forest":       "Random_Forest.joblib",
    "AdaBoost":            "AdaBoost.joblib",
    "Decision Tree":       "Decision_Tree.joblib",
}


def load_artefacts():
    """Load TF-IDF vectorizer and all trained models from disk."""
    tfidf = load_vectorizer(os.path.join(
        MODELS_DIR, "tfidf_vectorizer.joblib"))
    models = {}
    for name, fname in AVAILABLE_MODELS.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return tfidf, models


def load_sample_data(n: int = 5000) -> pd.DataFrame:
    """Load a slice of the training data for 'similar posts' lookup."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=['text', 'label', 'clean_text'])
    df = pd.read_csv(DATA_PATH)
    return df.head(n)


# ── Global state ──────────────────────────────────────────────────────────────
try:
    TFIDF, TRAINED_MODELS = load_artefacts()
    SAMPLE_DATA = load_sample_data()
    ARTEFACTS_LOADED = len(TRAINED_MODELS) > 0
except Exception as e:
    print(f"Warning: Could not load artefacts — {e}")
    TFIDF, TRAINED_MODELS = None, {}
    SAMPLE_DATA = pd.DataFrame()
    ARTEFACTS_LOADED = False


# ── Prediction function ───────────────────────────────────────────────────────

def predict_risk(user_input: str, model_name: str) -> tuple:
    """Main prediction function for Gradio interface."""

    if not user_input or not user_input.strip():
        return "Please enter a post.", 0.0, "", ""

    if not ARTEFACTS_LOADED:
        return "Models not loaded. Please run the training pipeline first.", 0.0, "", ""

    if model_name not in TRAINED_MODELS:
        return f"Model '{model_name}' not found.", 0.0, "", ""

    try:
        model = TRAINED_MODELS[model_name]

        # Pre-process
        clean = preprocess_text(user_input)
        clean_series = pd.Series([clean])
        raw_series = pd.Series([user_input])
        X = build_feature_matrix(clean_series, raw_series, TFIDF, fit=False)

        # Predict
        pred = model.predict(X)[0]

        # Confidence
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            confidence = float(proba[pred]) * 100
        elif hasattr(model, 'decision_function'):
            score = model.decision_function(X)[0]
            confidence = float(1 / (1 + np.exp(-score))) * 100
            if pred == 0:
                confidence = 100 - confidence
        else:
            confidence = 100.0 if pred == 1 else 0.0

        label = "Suicide Risk" if pred == 1 else "Non-Risk"

        # LLM explanation — api key handled internally by llm.py
        llm_explanation = explain_prediction(
            text=user_input,
            label=label,
            confidence=confidence,
        )

        # Similar posts
        similar_md = ""
        if not SAMPLE_DATA.empty and 'label' in SAMPLE_DATA.columns:
            same_class = SAMPLE_DATA[SAMPLE_DATA['label'] == pred]
            if len(same_class) >= 3:
                samples = same_class.sample(
                    min(3, len(same_class)), random_state=42)
                similar_md = "Similar posts from the training dataset:\n\n"
                for _, row in samples.iterrows():
                    snippet = str(row['text'])[:200].replace('\n', ' ')
                    similar_md += f"> {snippet}…\n\n"

        return label, round(confidence, 2), llm_explanation, similar_md

    except Exception as e:
        return f"Error: {str(e)}", 0.0, "", ""


# ── Gradio Interface ──────────────────────────────────────────────────────────

def build_interface():
    """Build and configure Gradio interface."""

    model_choices = list(TRAINED_MODELS.keys()) if TRAINED_MODELS else list(
        AVAILABLE_MODELS.keys())

    with gr.Blocks(title="Mental Health Post Detector") as demo:

        gr.Markdown("""
        # Suicide & Depression Post Detection System
        
        Using NLP, ML classification, and LLM integration
        
        ---
        
        Analyze posts using machine learning. Select a model and click Analyze.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                user_input = gr.Textbox(
                    label="Enter Post",
                    placeholder="Paste or type the post content here...",
                    lines=10,
                    max_lines=20,
                )

                model_choice = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0] if model_choices else None,
                    label="Select ML Model",
                )

                predict_btn = gr.Button(
                    "Analyze Post",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Results")

                prediction = gr.Textbox(
                    label="Prediction",
                    interactive=False,
                )

                confidence = gr.Number(
                    label="Confidence Score (%)",
                    interactive=False,
                )

                explanation = gr.Textbox(
                    label="LLM Explanation",
                    interactive=False,
                    lines=6,
                )

                similar = gr.Textbox(
                    label="Similar Posts",
                    interactive=False,
                )

        with gr.Accordion("Sample Posts", open=False):
            gr.Examples(
                examples=[
                    ["I have been struggling with depression and I don't think I can go on anymore. Everything feels hopeless.", "Logistic Regression"],
                    ["Had a great day today! Feeling optimistic about the future and excited about my projects.", "Random Forest"],
                    ["The pain is unbearable. I see no way forward.",
                        "SVM (LinearSVC)"],
                    ["Just finished a successful project at work. Feeling proud and accomplished.", "AdaBoost"],
                ],
                inputs=[user_input, model_choice],
            )

        gr.Markdown("""
        ---
        ### Support Resources
        If you are experiencing suicidal thoughts, please reach out to a mental health professional or crisis service immediately.
        Professional support is available 24/7.
        """)

        predict_btn.click(
            fn=predict_risk,
            inputs=[user_input, model_choice],
            outputs=[prediction, confidence, explanation, similar],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Mental Health Post Analysis System...")

    if not ARTEFACTS_LOADED:
        print("WARNING: Models not loaded. Please run the training pipeline first.")

    demo = build_interface()
    demo.launch(
        share=True,
        server_port=None,
        show_error=True,
    )
