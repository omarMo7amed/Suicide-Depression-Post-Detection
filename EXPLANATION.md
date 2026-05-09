# Suicide & Depression Post Detection — Plain-Language Explanation

## What is this project about?

This project builds a computer system that reads Reddit posts and tries to decide whether the person who wrote the post might be at risk of suicide or not. It uses text from real Reddit communities — specifically the "SuicideWatch" subreddit and other general discussion posts — to teach the computer the difference between someone who is in crisis and someone who is not.

---

## Why does this matter?

Every day, thousands of people reach out on the internet when they are in pain. Many of them post on Reddit before they ever call a hotline or speak to a doctor. If we can automatically detect these high-risk posts early, moderators, crisis counsellors, and mental health platforms could respond faster and potentially save lives.

---

## How does the computer read and understand text?

Computers don't understand language the way humans do. We have to convert words into numbers first. Here is how we do it step by step:

**Step 1 — Cleaning the text**
We remove all the noise: website links, punctuation, capital letters, digits. We keep only the meaningful words.

**Step 2 — Breaking it into pieces (Tokenisation)**
We split the sentence into individual words, called "tokens". For example, "I feel hopeless" becomes ["I", "feel", "hopeless"].

**Step 3 — Removing common words (Stopword Removal)**
Words like "the", "a", "is" appear everywhere and don't tell us much. We remove them — but we keep words like "not" and "never" because they matter a lot in mental-health text ("I do NOT want to live" means something very different without the "not").

**Step 4 — Simplifying words (Lemmatisation)**
"Feeling", "felt", "feels" all mean the same root idea. We simplify them all to "feel". This helps the computer recognise the same concept in different forms.

**Step 5 — Turning words into numbers (TF-IDF)**
We use a mathematical method called TF-IDF (Term Frequency–Inverse Document Frequency). It gives higher scores to words that appear often in one post but rarely across all posts — these distinctive words are the most informative.

---

## What extra clues do we use?

Beyond the words themselves, we also measure:

- **How long is the post?** Suicide-risk posts tend to be longer and more detailed.
- **How many "crisis words" appear?** Words like "hopeless", "worthless", "goodbye", "end it".
- **How often does the person say "I", "me", "my"?** High first-person usage often signals inward-focused distress.
- **Are there negation words?** "I can't", "I won't", "never" are powerful signals.
- **Is the writing frantic?** Lots of exclamation marks can indicate emotional dysregulation.

---

## How do the machine learning models work?

We train six different models — each learns from thousands of labelled examples and discovers its own pattern for spotting risk:

1. **Logistic Regression** — The simplest model. Learns a mathematical boundary between "safe" and "at-risk" posts.
2. **Support Vector Machine (SVM)** — Finds the widest possible gap between the two categories.
3. **Decision Tree** — Asks a series of yes/no questions (e.g. "Does the post contain 'hopeless'? → Yes → Does it contain 'end it'? → Yes → Label: Risk").
4. **Random Forest** — Builds hundreds of Decision Trees and takes the majority vote.
5. **AdaBoost** — Learns from its own mistakes: each new model focuses more on the examples the previous model got wrong.

We test all six models and pick the one that works best on data it has never seen before.

---

## How do we measure success?

Because missing a real crisis post is far worse than flagging a safe one by mistake, we care most about:

- **Recall** — Of all actual crisis posts, how many did we catch? (We want this as high as possible.)
- **Precision** — Of all posts we flagged, how many were actually in crisis?
- **F1-Score** — A balanced combination of both.
- **ROC-AUC** — How well can the model distinguish between the two groups overall?

---

## What does the AI explanation feature do?

After the ML model makes a prediction, we send the post to a powerful language model (Groq's LLaMA-3) and ask it to explain the decision in plain English. It points out which specific phrases triggered the classification and adds any nuance a human reviewer should consider. This makes the system more transparent and easier to trust.

---

## What are the limitations?

- The model can make mistakes — it may miss some crisis posts or flag some safe ones.
- It cannot replace a trained clinician or crisis counsellor.
- The model only understands text; it cannot hear tone of voice or see context.
- It was trained on Reddit data and may not generalise perfectly to other platforms.

---

## Who should use this?

This system is designed for **researchers, mental health platform developers, and content moderators** who want to build better early-warning tools. It should always be used alongside human review — never as a sole decision-maker.

---

_If you are personally struggling, please reach out to a mental health professional or a crisis helpline in your country._
