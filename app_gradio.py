"""
app_gradio.py
=============
Web interface for Mental Health Post Analysis System.
Provides an interactive dashboard for analysing posts using machine learning models
and optionally generating AI-powered explanations.

Features:
  - Text input for posts or messages
  - Multiple ML model selection
  - Real-time prediction with confidence scores
  - AI-powered analysis explanations via Groq
  - Related example posts from training data
  - Professional, modern user interface

Usage:
    python app_gradio.py

The application will launch a web server at http://127.0.0.1:7860
"""

from src.llm import explain_prediction
from src.features import build_feature_matrix, load_vectorizer, extract_handcrafted_features
from src.preprocessing import preprocess_text, download_nltk_resources
import os
import sys
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# ── Make sure src/ is importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Download NLTK data ────────────────────────────────────────────────────────
download_nltk_resources()

# ── Load artefacts ────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_PATH = "data/suicide_detection.csv"

AVAILABLE_MODELS = {
    "Logistic Regression":  "Logistic_Regression.joblib",
    "SVM (LinearSVC)":      "SVM_LinearSVC.joblib",
    "Random Forest":        "Random_Forest.joblib",
    "AdaBoost":             "AdaBoost.joblib",
    "Decision Tree":        "Decision_Tree.joblib",
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
    # Use pre-processed version if available
    clean_col = 'clean_text' if 'clean_text' in df.columns else df.columns[0]
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


# ── Prediction logic ──────────────────────────────────────────────────────────

def predict(text: str,
            model_name: str,
            groq_api_key: str,
            explain: bool) -> tuple:
    """
    Core prediction function called by the Gradio interface.

    Returns: (label_str, confidence_pct, explanation_md, similar_posts_md)
    """
    if not text or not text.strip():
        return "Please enter some text to analyse.", 0.0, "", ""

    if not ARTEFACTS_LOADED:
        return (
            "Models are not loaded. Please run the training pipeline first.",
            0.0, "", ""
        )

    if model_name not in TRAINED_MODELS:
        return f"Model '{model_name}' not found.", 0.0, "", ""

    model = TRAINED_MODELS[model_name]

    # ── Pre-process ────────────────────────────────────────────────────────
    clean = preprocess_text(text)
    clean_series = pd.Series([clean])
    raw_series = pd.Series([text])

    X = build_feature_matrix(clean_series, raw_series, TFIDF, fit=False)

    # ── Predict ───────────────────────────────────────────────────────────
    pred = model.predict(X)[0]

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        confidence = float(proba[pred]) * 100
    elif hasattr(model, 'decision_function'):
        score = model.decision_function(X)[0]
        # Sigmoid approximation
        confidence = float(1 / (1 + np.exp(-score))) * 100
        if pred == 0:
            confidence = 100 - confidence
    else:
        confidence = 100.0 if pred == 1 else 0.0

    label_str = "High Risk - Suicide Indicators Detected" if pred == 1 else "Low Risk - No Concerning Indicators"

    # ── LLM Explanation ───────────────────────────────────────────────────
    explanation_md = ""
    if explain:
        label_clean = "Suicide Risk" if pred == 1 else "Non-Suicide"
        explanation_md = explain_prediction(
            text=text,
            label=label_clean,
            confidence=confidence,
            api_key=groq_api_key or None,
        )

    # ── Similar Posts ─────────────────────────────────────────────────────
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

    return label_str, round(confidence, 2), explanation_md, similar_md


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
body { 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    min-height: 100vh;
}
.gr-container {
    max-width: 1000px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
}
.gr-button-primary { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
}
.gr-textbox, .gr-dropdown, .gr-slider {
    border-radius: 8px !important;
    border-color: #e0e0e0 !important;
}
.gr-textbox:focus, .gr-dropdown:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}
.gr-markdown {
    line-height: 1.6;
    color: #333;
}
.prediction-high-risk {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
    color: white;
    padding: 20px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1.1em;
}
.prediction-low-risk {
    background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
    color: white;
    padding: 20px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1.1em;
}
.gr-slider > .gr-slider-container {
    border-radius: 8px;
}
"""


def build_interface():
    model_choices = list(TRAINED_MODELS.keys()) if TRAINED_MODELS else list(
        AVAILABLE_MODELS.keys())

    with gr.Blocks(css=CSS, title="Mental Health Post Detector") as demo:

        gr.Markdown(
            """
            # Mental Health Post Analysis System
            #### Powered by NLP, Machine Learning, and AI
            
            This tool analyses posts using advanced machine learning models trained on real Reddit data.
            It identifies linguistic patterns associated with suicide risk to support research and moderation.
            
            **Important:** This system is designed for research and support purposes only. 
            It is not a substitute for professional mental health assessment or crisis intervention.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Enter Post Content")
                text_input = gr.Textbox(
                    label="Post Text",
                    placeholder="Paste or type the content you want to analyse...",
                    lines=8,
                    max_lines=20,
                )

                gr.Markdown("### Analysis Settings")
                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[0] if model_choices else None,
                        label="Select ML Model",
                    )
                    explain_toggle = gr.Checkbox(
                        label="Get AI Explanation",
                        value=False,
                    )

                groq_key = gr.Textbox(
                    label="API Key for AI Explanation",
                    placeholder="Enter your API key...",
                    type="password",
                    visible=False,
                )
                explain_toggle.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=explain_toggle,
                    outputs=groq_key,
                )

                predict_btn = gr.Button(
                    "Analyse Content", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                label_out = gr.Textbox(
                    label="Classification Result",
                    interactive=False,
                    elem_classes="output-label",
                )
                conf_out = gr.Slider(
                    label="Confidence Score (%)",
                    minimum=0, maximum=100,
                    interactive=False,
                )
                explanation_out = gr.Markdown(label="Detailed Analysis")
                similar_out = gr.Markdown(label="Related Examples")

        with gr.Accordion("Sample Posts for Testing", open=False):
            gr.Examples(
                examples=[
                    ["I have been struggling with depression for so long and I don't think I can continue anymore. Everything feels hopeless and empty.",
                        "Logistic Regression", False, ""],
                    ["Today was wonderful! Had a great time with family and feeling optimistic about the future.",
                        "Random Forest", False, ""],
                    ["The pain is unbearable. I cannot see any way forward and I have decided to end my suffering permanently.",
                        "SVM (LinearSVC)", True, ""],
                    ["Just finished a great project at work and feeling accomplished and proud of my efforts.",
                        "AdaBoost", False, ""],
                ],
                inputs=[text_input, model_choice, explain_toggle, groq_key],
            )

        gr.Markdown(
            """
            ---
            ### Support Resources
            If you or someone you know is experiencing suicidal thoughts, please reach out to a mental health professional or crisis service immediately. Professional support is available 24/7.
            """
        )

        predict_btn.click(
            fn=predict,
            inputs=[text_input, model_choice, groq_key, explain_toggle],
            outputs=[label_out, conf_out, explanation_out, similar_out],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Mental Health Post Analysis System...")
    print("Finding available port and launching interface...")
    demo = build_interface()
    demo.launch(
        share=True,           # Creates a public link
        server_port=None,     # Auto-find available port
        show_error=True,
    )
