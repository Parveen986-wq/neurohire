"""Resume text extraction (PDF via PyMuPDF, DOCX via python-docx) and section parsing."""
import io
import re

import fitz  # PyMuPDF
import docx

from .skills import SKILLS

SECTION_HEADERS = {
    "education": ["education", "academic", "qualification"],
    "experience": ["experience", "employment", "work history", "internship"],
    "projects": ["projects", "project work", "academic projects"],
    "skills": ["skills", "technical skills", "technologies", "tools"],
}


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    if name.endswith(".docx"):
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Upload PDF or DOCX.")


def extract_skills(text: str) -> list[str]:
    low = text.lower()
    found = []
    for skill in SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, low):
            found.append(skill)
    return sorted(set(found))


def split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {k: [] for k in SECTION_HEADERS}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower().rstrip(":")
        hit = next((k for k, keys in SECTION_HEADERS.items()
                    if len(low) < 40 and any(low.startswith(h) for h in keys)), None)
        if hit:
            current = hit
            continue
        if current:
            sections[current].append(line)
    return {k: v[:12] for k, v in sections.items()}


def extract_name(text: str, nlp=None) -> str:
    head = "\n".join(text.strip().splitlines()[:6])
    if nlp is not None:
        for ent in nlp(head).ents:
            if ent.label_ == "PERSON":
                return ent.text
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:60]


def extract_contact(text: str) -> dict:
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    phone = re.search(r"\+?\d[\d\s\-()]{8,}\d", text)
    return {"email": email.group(0) if email else None,
            "phone": phone.group(0).strip() if phone else None}


def years_of_experience(text: str) -> float:
    m = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", text.lower())
    return max((float(x) for x in m), default=0.0)
