from transformers import pipeline
from .preprocess import normalize_text, split_list_field

SPECIALTIES = ["Cardiology","General Practice","Pediatrics","Internal Medicine",
               "Neurology","Orthopedics","Radiology","Dermatology","Anesthesiology","Emergency Medicine"]

clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_specialization(text: str) -> str:
    res = clf(normalize_text(text)[:2000], SPECIALTIES)
    return res["labels"][0], float(res["scores"][0])

def competency_match(job_skills_str: str, cv_text: str) -> float:
    # zero-shot each must-have skill as a label (cheap proxy for skill relevance)
    labels = split_list_field(job_skills_str)
    if not labels: return 0.0
    res = clf(normalize_text(cv_text)[:2000], labels)
    # take mean of matched label scores
    return float(sum(res["scores"]) / len(res["scores"]))
