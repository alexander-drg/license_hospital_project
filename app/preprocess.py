import re

def normalize_text(x: str) -> str:
    x = x.lower().strip()
    x = re.sub(r"[^a-z0-9\s\-+/.,]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x

def split_list_field(s: str):
    return [t.strip().lower() for t in s.split(",") if t.strip()]

def skill_overlap(job_must, cand_skills):
    if not job_must: return 0.0
    j = set(job_must); c = set(cand_skills)
    return len(j & c) / max(1, len(j))

import re
import unicodedata


RO_EN_RECRUITMENT_TERMS = {
    # Medical / healthcare
    "medic": "doctor physician medical",
    "doctor": "doctor physician medical",
    "asistent medical": "nurse medical assistant healthcare",
    "asistenta medicala": "nurse medical assistant healthcare",
    "asistentă medicală": "nurse medical assistant healthcare",
    "spital": "hospital healthcare clinic",
    "clinica": "clinic healthcare medical",
    "clinică": "clinic healthcare medical",
    "cabinet medical": "medical office clinic healthcare",
    "pacient": "patient healthcare medical",
    "pacienti": "patients healthcare medical",
    "pacienți": "patients healthcare medical",
    "fisa pacient": "patient record medical record",
    "fișa pacient": "patient record medical record",
    "fise pacienti": "patient records medical records",
    "fișe pacienți": "patient records medical records",
    "dosare pacienti": "patient records medical records",
    "dosare pacienți": "patient records medical records",
    "programari": "appointments scheduling",
    "programări": "appointments scheduling",
    "programari pacienti": "patient appointments scheduling",
    "programări pacienți": "patient appointments scheduling",
    "administratie medicala": "healthcare administration medical administration",
    "administrație medicală": "healthcare administration medical administration",
    "secretar medical": "medical secretary healthcare administration",
    "receptie medicala": "medical reception healthcare administration",
    "recepție medicală": "medical reception healthcare administration",
    "registrator medical": "medical registrar healthcare administration",
    "documente medicale": "medical documentation medical records",
    "rapoarte medicale": "medical reports clinical documentation",
    "trimiteri": "referrals medical referrals",
    "recomandari medicale": "medical referrals recommendations",

    # Imaging / radiology
    "radiologie": "radiology imaging medical",
    "radiolog": "radiologist radiology doctor imaging",
    "imagistica": "imaging radiology diagnostics",
    "imagistică": "imaging radiology diagnostics",
    "centru de imagistica": "imaging centre radiology clinic diagnostics",
    "centru de imagistică": "imaging centre radiology clinic diagnostics",
    "ecografie": "ultrasound imaging diagnostics",
    "ecografii": "ultrasound imaging diagnostics",
    "tomografie": "ct computed tomography imaging",
    "ct": "ct computed tomography imaging",
    "rmn": "mri magnetic resonance imaging",
    "irm": "mri magnetic resonance imaging",
    "rezonanta magnetica": "mri magnetic resonance imaging",
    "rezonanță magnetică": "mri magnetic resonance imaging",
    "radiografie": "x-ray radiography imaging",
    "radiografii": "x-ray radiography imaging",
    "pacs": "pacs radiology imaging system",
    "dicom": "dicom radiology imaging system",

    # Cardiology
    "cardiologie": "cardiology medical doctor",
    "cardiolog": "cardiologist cardiology doctor",
    "ecg": "ecg electrocardiogram cardiology",
    "ekg": "ecg electrocardiogram cardiology",
    "electrocardiograma": "ecg electrocardiogram cardiology",
    "electrocardiogramă": "ecg electrocardiogram cardiology",
    "hipertensiune": "hypertension cardiology medical",
    "insuficienta cardiaca": "heart failure cardiology medical",
    "insuficiență cardiacă": "heart failure cardiology medical",
    "consultatii ambulatorii": "outpatient consultations medical",
    "consultații ambulatorii": "outpatient consultations medical",

    # General job terms
    "experienta": "experience",
    "experiență": "experience",
    "ani experienta": "years experience",
    "ani experiență": "years experience",
    "responsabilitati": "responsibilities",
    "responsabilități": "responsibilities",
    "cerinte": "requirements",
    "cerințe": "requirements",
    "abilitati": "skills",
    "abilități": "skills",
    "competente": "competencies skills",
    "competențe": "competencies skills",
    "studii": "education",
    "educatie": "education",
    "educație": "education",
    "limbi straine": "foreign languages",
    "limbi străine": "foreign languages",
    "norma intreaga": "full-time",
    "normă întreagă": "full-time",
    "program": "schedule",
    "locatie": "location city",
    "locație": "location city",

    # Sales / marketing, useful for unrelated detection
    "vanzari": "sales",
    "vânzări": "sales",
    "agent vanzari": "sales representative",
    "agent vânzări": "sales representative",
    "reprezentant vanzari": "sales representative",
    "reprezentant vânzări": "sales representative",
    "marketing": "marketing",
    "marketing digital": "digital marketing",
    "campanii": "campaigns marketing",
    "campanii publicitare": "advertising campaigns marketing",
    "social media": "social media marketing",
    "seo": "seo digital marketing",
    "lead generation": "lead generation sales marketing",
    "clienti": "clients customers",
    "clienți": "clients customers",
    "vanzare": "sales selling",
    "vânzare": "sales selling",
}


def strip_accents(text: str) -> str:
    if not text:
        return ""

    normalized = unicodedata.normalize("NFD", text)
    without_accents = "".join(
        ch for ch in normalized
        if unicodedata.category(ch) != "Mn"
    )

    return without_accents


def normalize_ro_en_text(text: str) -> str:
    """
    Expands Romanian recruitment/medical terms into English equivalents.
    This helps matching when a CV is in English and the job description is in Romanian,
    or the other way around.
    """
    if not text:
        return ""

    original = text
    lower = text.lower()
    lower_no_accents = strip_accents(lower)

    additions = []

    for ro_term, en_terms in RO_EN_RECRUITMENT_TERMS.items():
        ro_l = ro_term.lower()
        ro_l_no_accents = strip_accents(ro_l)

        if ro_l in lower or ro_l_no_accents in lower_no_accents:
            additions.append(en_terms)

    if additions:
        return original + "\n\nNormalized bilingual terms: " + " ".join(sorted(set(additions)))

    return original