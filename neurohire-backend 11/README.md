# NeuroHire - AI Resume Analyzer (Python backend)

FastAPI service using PyMuPDF, python-docx, spaCy, TF-IDF + cosine similarity,
Sentence-Transformers, a Keras neural network role classifier and a logistic-regression
shortlist model.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app.main:app --reload --port 8000
```

First start trains the neural network (~30s) and saves it to `models/`.
Open http://localhost:8000/docs for interactive API docs.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | /health | status |
| POST | /analyze | multipart: `resume` (PDF/DOCX), `job_description` (text) |
| GET | /model/metrics | training accuracy/loss per epoch |
| POST | /model/retrain?epochs=15 | retrain models |

## Optional Supabase storage
Run `schema.sql` in your database, then set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`.

## ML pipeline
1. Extract text -> regex/spaCy for name, contact, sections, skills.
2. TF-IDF cosine similarity (lexical) + MiniLM embeddings (semantic) vs. job description.
3. Dense NN (256->128->softmax, dropout 0.3) predicts best-fit job roles.
4. Logistic regression turns features into a shortlist probability.
5. Rules generate suggestions and a learning path for missing skills.

Training data is synthetic (skill sampling per role); swap `_synthetic_corpus` in
`app/ml.py` for a real labelled dataset (e.g. Kaggle Resume Dataset) for better accuracy.
