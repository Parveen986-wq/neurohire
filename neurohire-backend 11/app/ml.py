"""Machine learning core.

* TF-IDF + cosine similarity  -> lexical resume/JD match
* Sentence-Transformers        -> semantic resume/JD match
* Keras neural network          -> job role classification (multi-class, softmax)
* Logistic regression          -> shortlist probability from engineered features
"""
from __future__ import annotations

import os
import random

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from .skills import ROLES, SKILLS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def tfidf_similarity(a: str, b: str) -> float:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    m = vec.fit_transform([a, b])
    return float(cosine_similarity(m[0], m[1])[0][0])


def semantic_similarity(a: str, b: str) -> float:
    emb = embedder().encode([a, b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


# ---------------- Neural network role classifier ----------------

def _synthetic_corpus(n_per_role: int = 300):
    """Builds a synthetic training set by sampling skill subsets per role.
    Replace with a real labelled resume dataset (e.g. Kaggle 'Resume Dataset') for production."""
    rng = random.Random(42)
    texts, labels = [], []
    for role, skills in ROLES.items():
        for _ in range(n_per_role):
            k = rng.randint(max(3, len(skills) // 2), len(skills))
            picked = rng.sample(skills, k) + rng.sample(SKILLS, 3)  # noise
            texts.append(" ".join(picked) + f" {role.lower()}")
            labels.append(role)
    return texts, labels


class RoleClassifier:
    def __init__(self):
        self.vec_path = os.path.join(MODEL_DIR, "role_tfidf.joblib")
        self.enc_path = os.path.join(MODEL_DIR, "role_labels.joblib")
        self.nn_path = os.path.join(MODEL_DIR, "role_nn.keras")
        self.vectorizer = None
        self.encoder = None
        self.model = None
        self.history = None

    def train(self, epochs: int = 15):
        from tensorflow import keras
        texts, labels = _synthetic_corpus()
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts).toarray()
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(labels)
        n = len(self.encoder.classes_)
        self.model = keras.Sequential([
            keras.layers.Input(shape=(X.shape[1],)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(n, activation="softmax"),
        ])
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = self.model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
        self.history = {k: [float(x) for x in v] for k, v in h.history.items()}
        self.model.save(self.nn_path)
        joblib.dump(self.vectorizer, self.vec_path)
        joblib.dump(self.encoder, self.enc_path)
        joblib.dump(self.history, os.path.join(MODEL_DIR, "role_history.joblib"))

    def load(self):
        from tensorflow import keras
        if not os.path.exists(self.nn_path):
            self.train()
            return
        self.model = keras.models.load_model(self.nn_path)
        self.vectorizer = joblib.load(self.vec_path)
        self.encoder = joblib.load(self.enc_path)
        hp = os.path.join(MODEL_DIR, "role_history.joblib")
        self.history = joblib.load(hp) if os.path.exists(hp) else None

    def predict(self, text: str, top_k: int = 4) -> list[dict]:
        X = self.vectorizer.transform([text]).toarray()
        probs = self.model.predict(X, verbose=0)[0]
        order = np.argsort(probs)[::-1][:top_k]
        return [{"role": self.encoder.classes_[i], "confidence": round(float(probs[i]) * 100, 1)}
                for i in order]


# ---------------- Shortlist probability (logistic regression) ----------------

class ShortlistModel:
    """Features: [skill_match, tfidf_sim, semantic_sim, years_exp_norm, n_projects_norm]"""

    def __init__(self):
        self.path = os.path.join(MODEL_DIR, "shortlist_lr.joblib")
        self.model: LogisticRegression | None = None

    def train(self):
        rng = np.random.default_rng(7)
        X = rng.random((2000, 5))
        logits = 4 * X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2] + 1.2 * X[:, 3] + 0.8 * X[:, 4] - 5.2
        y = (rng.random(2000) < 1 / (1 + np.exp(-logits))).astype(int)
        self.model = LogisticRegression().fit(X, y)
        joblib.dump(self.model, self.path)

    def load(self):
        if os.path.exists(self.path):
            self.model = joblib.load(self.path)
        else:
            self.train()

    def predict(self, feats: list[float]) -> float:
        return float(self.model.predict_proba([feats])[0][1])
