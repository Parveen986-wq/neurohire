"""NeuroHire - AI Resume Analyzer API (FastAPI)."""
from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import extractor
from .db import save_analysis
from .ml import RoleClassifier, ShortlistModel, semantic_similarity, tfidf_similarity
from .skills import LEARNING, ROLES, SOFT_SKILLS

app = FastAPI(title="NeuroHire API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

roles = RoleClassifier()
shortlist = ShortlistModel()
nlp = None


@app.on_event("startup")
def startup():
    global nlp
    roles.load()
    shortlist.load()
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None


@app.get("/health")
def health():
    return {"status": "ok", "spacy": nlp is not None}


@app.get("/model/metrics")
def metrics():
    return {"history": roles.history, "classes": list(roles.encoder.classes_)}


@app.post("/model/retrain")
def retrain(epochs: int = 15):
    roles.train(epochs=epochs)
    shortlist.train()
    return {"status": "retrained", "final_val_accuracy": roles.history["val_accuracy"][-1]}


@app.post("/analyze")
async def analyze(resume: UploadFile = File(...), job_description: str = Form("")):
    data = await resume.read()
    try:
        text = extractor.extract_text(resume.filename or "", data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if len(text.strip()) < 50:
        raise HTTPException(400, "Could not read enough text from this resume.")

    skills = extractor.extract_skills(text)
    sections = extractor.split_sections(text)
    years = extractor.years_of_experience(text)
    predicted = roles.predict(text)

    jd = job_description.strip()
    if jd:
        jd_skills = extractor.extract_skills(jd)
    else:
        jd_skills = ROLES[predicted[0]["role"]]
        jd = " ".join(jd_skills)

    matched = [s for s in jd_skills if s in skills]
    missing = [s for s in jd_skills if s not in skills]
    skill_match = len(matched) / max(len(jd_skills), 1)
    tfidf = tfidf_similarity(text, jd)
    semantic = semantic_similarity(text[:4000], jd[:4000])
    n_projects = len(sections["projects"])

    prob = shortlist.predict([skill_match, tfidf, max(semantic, 0), min(years / 10, 1), min(n_projects / 8, 1)])
    ats = round(100 * (0.45 * skill_match + 0.2 * tfidf + 0.35 * max(semantic, 0)), 1)

    category_scores = {
        "Technical": round(100 * skill_match, 1),
        "Semantic Fit": round(100 * max(semantic, 0), 1),
        "Keywords": round(100 * min(tfidf * 2, 1), 1),
        "Experience": round(min(years / 5, 1) * 100, 1),
        "Projects": round(min(n_projects / 5, 1) * 100, 1),
        "Soft Skills": round(100 * len([s for s in skills if s in SOFT_SKILLS]) / len(SOFT_SKILLS), 1),
    }

    suggestions = []
    if missing:
        suggestions.append(f"Add or learn these job-relevant skills: {', '.join(missing[:6])}.")
    if not sections["projects"]:
        suggestions.append("Add a Projects section with 2-3 quantified projects.")
    if tfidf < 0.15:
        suggestions.append("Mirror key phrases from the job description to improve ATS keyword match.")
    if years == 0:
        suggestions.append("State your experience duration explicitly (e.g. '2 years').")
    if category_scores["Soft Skills"] < 40:
        suggestions.append("Show soft skills like leadership or teamwork through concrete examples.")

    result = {
        "candidate": {"name": extractor.extract_name(text, nlp), **extractor.extract_contact(text),
                      "years_experience": years},
        "sections": sections,
        "skills": skills,
        "jd_skills": jd_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "scores": {
            "ats_score": ats,
            "skill_match": round(skill_match * 100, 1),
            "tfidf_similarity": round(tfidf * 100, 1),
            "semantic_similarity": round(max(semantic, 0) * 100, 1),
            "shortlist_probability": round(prob * 100, 1),
        },
        "category_scores": category_scores,
        "predicted_roles": predicted,
        "suggestions": suggestions,
        "learning_path": [{"skill": s, "resource": LEARNING.get(s, f"Search: '{s} beginner course'")}
                          for s in missing[:6]],
        "text_preview": text[:1500],
    }
    save_analysis(resume.filename, result)
    return result
