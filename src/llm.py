
import os
import time
from groq import Groq

# ── Configuration ────────────────────────────────────────────────────────────

GROQ_API_KEY = "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"
MODEL_NAME = "mixtral-8x7b-32768"


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a compassionate mental health AI assistant helping clinicians 
and researchers understand why text posts are flagged as suicide-risk or non-risk.

Your role:
1. Analyse the linguistic signals in the text
2. Explain the prediction clearly and professionally
3. Highlight specific phrases or patterns that influenced the classification
4. Always maintain a caring, non-judgmental tone
5. If the post is classified as suicide-risk, include a brief note on potential support

Keep your response to 3–4 short paragraphs. Do NOT make clinical diagnoses."""

USER_PROMPT_TEMPLATE = """Analyse the following Reddit post and explain why the model classified it as shown below.

POST:
\"\"\"{text}\"\"\"

MODEL OUTPUT:
- Prediction  : {label}
- Confidence  : {confidence:.1f}%

Please explain:
1. Which linguistic signals (words, tone, themes) led to this prediction
2. Whether the confidence level seems appropriate given the text
3. Any nuances or caveats a human reviewer should consider
{support_note}"""

SUPPORT_NOTE_SUICIDE = """
4. Briefly mention that professional support resources exist (e.g., crisis hotlines) 
   without listing specific numbers."""

SUPPORT_NOTE_NORMAL = ""


# ── Main function ─────────────────────────────────────────────────────────────

def explain_prediction(text: str,
                       label: str,
                       confidence: float,
                       api_key: str = None,
                       max_retries: int = 3,
                       model_name: str = None) -> str:

    key = api_key or GROQ_API_KEY
    if not key:
        return (
            "⚠️ Groq API key not set. "
            "Please set the GROQ_API_KEY environment variable to enable LLM explanations."
        )

    client = Groq(api_key=key)

    support_note = SUPPORT_NOTE_SUICIDE if label == 'Suicide Risk' else SUPPORT_NOTE_NORMAL

    user_prompt = USER_PROMPT_TEMPLATE.format(
        text=text[:2000],           # Truncate very long posts
        label=label,
        confidence=confidence,
        support_note=support_note,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=600,
                temperature=0.4,
            )
            explanation = response.choices[0].message.content.strip()
            return explanation

        except Exception as e:
            error_str = str(e).lower()
            if 'rate' in error_str and attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"Rate limit hit. Waiting {wait}s before retry {attempt}/{max_retries}…")
                time.sleep(wait)
            else:
                return f"LLM explanation unavailable: {str(e)}"

    return "LLM explanation unavailable after multiple retries."


# ── Batch explanation helper ──────────────────────────────────────────────────

def batch_explain(texts: list,
                  labels: list,
                  confidences: list,
                  api_key: str = None) -> list:

    explanations = []
    for text, label, conf in zip(texts, labels, confidences):
        exp = explain_prediction(text, label, conf, api_key=api_key)
        explanations.append(exp)
        time.sleep(0.5)    # polite rate-limiting
    return explanations


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sample_text = (
        "I've been feeling completely worthless lately. "
        "Nothing I do matters. I keep thinking about ending everything. "
        "I don't know how much longer I can go on like this."
    )
    result = explain_prediction(
        text=sample_text,
        label='Suicide Risk',
        confidence=94.2,
    )
    print(result)
