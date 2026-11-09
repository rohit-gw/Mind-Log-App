"""
MindLog Custom Model — Trainer
Trains two components from your curated dataset:
  1. IntentClassifier  — TF-IDF + Logistic Regression
  2. ResponseRetriever — TF-IDF vectorizer over all responses

Both are saved to disk (joblib) so they only need training once.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
DATA_FILE   = HERE / "data" / "conversations.json"
MODEL_DIR   = HERE / "saved_model"
MODEL_DIR.mkdir(exist_ok=True)

INTENT_MODEL_PATH   = MODEL_DIR / "intent_classifier.joblib"
RESPONSE_VEC_PATH   = MODEL_DIR / "response_vectorizer.joblib"
RESPONSE_MATRIX_PATH = MODEL_DIR / "response_matrix.npy"
RESPONSE_MAP_PATH   = MODEL_DIR / "response_map.joblib"


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset() -> tuple[list[str], list[str], dict[str, list[str]]]:
    """
    Returns:
        patterns  : list of all pattern strings
        labels    : corresponding intent tag for each pattern
        responses : dict of {tag: [response strings]}
    """
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)

    patterns, labels = [], []
    responses: dict[str, list[str]] = {}

    for intent in data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]
        for p in intent["patterns"]:
            patterns.append(p.lower())
            labels.append(tag)

    return patterns, labels, responses


# ── Train Intent Classifier ───────────────────────────────────────────────────
def train_intent_classifier(patterns: list[str], labels: list[str]) -> Pipeline:
    """
    TF-IDF + Logistic Regression pipeline.
    Prints train/test accuracy report.
    """
    print("\n[1/2] Training Intent Classifier…")

    X_train, X_test, y_train, y_test = train_test_split(
        patterns, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            C=4.0,
            max_iter=1000,
            multi_class="auto",
            solver="lbfgs",
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nClassification Report (test set):")
    print(classification_report(y_test, y_pred, zero_division=0))

    train_acc = pipeline.score(X_train, y_train)
    test_acc  = pipeline.score(X_test, y_test)
    print(f"Train accuracy : {train_acc:.2%}")
    print(f"Test  accuracy : {test_acc:.2%}")

    return pipeline


# ── Build Response Retriever ──────────────────────────────────────────────────
def build_response_retriever(
    responses: dict[str, list[str]]
) -> tuple[TfidfVectorizer, np.ndarray, list[tuple[str, str]]]:
    """
    Vectorise all responses using TF-IDF.
    Returns (vectorizer, response_matrix, response_map).
    response_map[i] = (tag, response_text)
    """
    print("\n[2/2] Building Response Retriever…")

    response_map: list[tuple[str, str]] = []
    all_texts: list[str] = []

    for tag, resp_list in responses.items():
        for resp in resp_list:
            response_map.append((tag, resp))
            all_texts.append(resp.lower())

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(all_texts).toarray()

    print(f"  Indexed {len(response_map)} responses across {len(responses)} intents.")
    return vectorizer, matrix, response_map


# ── Save all artifacts ────────────────────────────────────────────────────────
def save_artifacts(
    intent_pipeline:  Pipeline,
    resp_vectorizer:  TfidfVectorizer,
    resp_matrix:      np.ndarray,
    resp_map:         list[tuple[str, str]],
) -> None:
    joblib.dump(intent_pipeline, INTENT_MODEL_PATH)
    joblib.dump(resp_vectorizer,  RESPONSE_VEC_PATH)
    np.save(RESPONSE_MATRIX_PATH, resp_matrix)
    joblib.dump(resp_map,          RESPONSE_MAP_PATH)
    print(f"\nAll artifacts saved to: {MODEL_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────
def train() -> None:
    print("=" * 60)
    print("  MindLog — Custom Chatbot Model Trainer")
    print("=" * 60)

    patterns, labels, responses = load_dataset()
    print(f"\nDataset loaded: {len(patterns)} patterns | {len(set(labels))} intents")

    intent_pipeline = train_intent_classifier(patterns, labels)
    resp_vec, resp_matrix, resp_map = build_response_retriever(responses)
    save_artifacts(intent_pipeline, resp_vec, resp_matrix, resp_map)

    print("\nTraining complete! Model is ready to use.")
    print("=" * 60)


if __name__ == "__main__":
    train()
