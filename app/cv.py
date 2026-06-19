# app/cv.py
import io, uuid, re
from typing import List, Tuple
from pypdf import PdfReader
import docx2txt
from PIL import Image
import pytesseract
import dateparser
import fitz
import pdfplumber

from .ollama_ai import analyze_cv_full, is_online
    

# ------- FILE READERS --------
def _read_pdf(data: bytes) -> str:
    """
    Reads PDF text using multiple extraction methods.
    Tries PyMuPDF first, then pdfplumber, then pypdf fallback.
    This improves extraction from CVs with complex layouts.
    """
    texts = []

    # 1. PyMuPDF extraction - usually better for structured CVs
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        for page in doc:
            text = page.get_text("text") or ""
            if text.strip():
                texts.append(text)
        doc.close()

        full_text = "\n".join(texts).strip()
        if len(full_text) > 100:
            return full_text
    except Exception:
        pass

    # 2. pdfplumber fallback - useful for tables / layout-heavy PDFs
    try:
        texts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                if text.strip():
                    texts.append(text)

        full_text = "\n".join(texts).strip()
        if len(full_text) > 100:
            return full_text
    except Exception:
        pass

    # 3. pypdf fallback - current method
    try:
        texts = []
        reader = PdfReader(io.BytesIO(data))
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                texts.append(text)
        return "\n".join(texts).strip()
    except Exception:
        return ""

def _read_docx(data: bytes) -> str:
    return docx2txt.process(io.BytesIO(data)) or ""

def _read_image(data: bytes) -> str:
    img = Image.open(io.BytesIO(data))
    return pytesseract.image_to_string(img)

def read_any(file_storage) -> str:
    data = file_storage.read()
    name = (file_storage.filename or "").lower()

    if name.endswith(".pdf"):
        return _read_pdf(data)
    if name.endswith(".docx") or name.endswith(".doc"):
        return _read_docx(data)
    if name.endswith(".txt"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    if any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
        return _read_image(data)

    try:
        return _read_pdf(data)
    except Exception:
        return ""
    
# ------- BASIC FIELDS --------
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,})")
CITIES = [
    "Bucharest","Cluj","Iași","Timisoara","Constanța","Brașov","Sibiu","Craiova",
    "Oradea","Galați","Ploiești","Bacău","Arad","Pitesti","Târgu Mureș","Botoșani",
    "Bistrița","Suceava","Buzău","Brăila","Focșani"
]
CERT_KEYWORDS = ["ACLS","BLS","ATLS","PALS","ALS","Neonatal Resuscitation","Radiology Certificate","GMP"]
SKILL_VOCAB = [
    "ecg","echocardiography","hypertension","primary care","vaccination","diagnostics",
    "ultrasound","x-ray","ct","mri","endoscopy","icu","emergency care","triage","telemedicine",
    "suturing","venipuncture","phlebotomy","spirometry","wound care","injections","blood sampling",
    "ehr","emr","billing","scheduling","ms office","outlook","access","telehealth"
]
SPECIALTIES = ["Cardiology","General Practice","Pediatrics","Internal Medicine","Neurology",
               "Orthopedics","Radiology","Dermatology","Anesthesiology","Emergency Medicine","Healthcare Administration"]

MEDICAL_STRONG_KEYWORDS = [
    # English
    "doctor", "physician", "nurse", "medical", "medicine", "healthcare",
    "clinic", "hospital", "patient", "patients", "diagnosis", "treatment",
    "radiology", "radiologist", "cardiology", "cardiologist", "pediatrics",
    "dermatology", "neurology", "orthopedics", "surgery", "pharmacy",
    "pharmacist", "dentist", "dental", "laboratory", "clinical",
    "imaging", "ultrasound", "x-ray", "xray", "ct", "mri", "pacs", "dicom",
    "medical secretary", "medical administrative assistant",
    "healthcare administrator", "health administration",
    "patient records", "medical records", "appointment scheduling",
    "referrals", "medical reports",

    # Romanian without accents
    "medic", "doctor", "asistent medical", "asistenta medicala",
    "infirmier", "farmacist", "farmacie", "spital", "clinica",
    "cabinet medical", "pacient", "pacienti", "diagnostic", "tratament",
    "consultatie", "consultatii", "vaccinare", "radiologie", "radiolog",
    "cardiologie", "cardiolog", "pediatrie", "dermatologie", "neurologie",
    "ortopedie", "chirurgie", "stomatologie", "dentist", "laborator",
    "imagistica", "centru de imagistica", "ecografie", "ecografii",
    "tomografie", "rezonanta magnetica", "radiografie", "radiografii",
    "administratie medicala", "secretar medical", "receptie medicala",
    "registrator medical", "fise pacienti", "dosare pacienti",
    "programari pacienti", "documente medicale", "rapoarte medicale",

    # Romanian with accents
    "asistentă medicală", "clinică", "pacienți", "consultație",
    "consultații", "imagistică", "centru de imagistică",
    "rezonanță magnetică", "administrație medicală", "recepție medicală",
    "fișe pacienți", "programări pacienți",
]


MEDICAL_WEAK_KEYWORDS = [
    # English
    "administration", "administrator", "office management", "records",
    "scheduling", "appointments", "confidentiality", "client relations",
    "customer service", "workflow", "documentation",

    # Romanian
    "administratie", "administrație", "administrator", "management",
    "programari", "programări", "documente", "documentatie", "documentație",
    "evidenta", "evidență", "confidentialitate", "confidențialitate",
    "relatii clienti", "relații clienți",
]

NON_MEDICAL_PROFESSION_KEYWORDS = [
    # English
    "sales", "marketing", "digital marketing", "business development",
    "account manager", "sales representative", "sales executive",
    "lead generation", "campaign management", "seo", "social media",
    "brand strategy", "advertising", "e-commerce", "crm",
    "customer acquisition", "market research",

    # Romanian
    "vanzari", "vânzări", "agent vanzari", "agent vânzări",
    "reprezentant vanzari", "reprezentant vânzări",
    "marketing digital", "campanii publicitare", "social media",
    "strategie de brand", "publicitate", "comert electronic",
    "comerț electronic", "achizitie clienti", "achiziție clienți",
    "cercetare de piata", "cercetare de piață",
]


def has_clear_non_medical_profession(text: str) -> bool:
    if not text:
        return False

    text_l = text.lower()
    hits = 0

    for kw in NON_MEDICAL_PROFESSION_KEYWORDS:
        if kw.lower() in text_l:
            hits += 1

    return hits >= 2


def medical_relevance_score(text: str) -> float:
    """
    Returns a medical relevance score between 0 and 1.
    Strong medical indicators matter much more than generic admin words.
    """
    if not text:
        return 0.0

    text_l = text.lower()

    strong_hits = 0
    weak_hits = 0

    for kw in MEDICAL_STRONG_KEYWORDS:
        if kw.lower() in text_l:
            strong_hits += 1

    for kw in MEDICAL_WEAK_KEYWORDS:
        if kw.lower() in text_l:
            weak_hits += 1

    if strong_hits == 0:
        return 0.0

    score = (strong_hits * 0.25) + (weak_hits * 0.05)

    return min(1.0, score)


def is_medical_related_cv(text: str) -> bool:
    score = medical_relevance_score(text)
    return score >= 0.25


def is_medical_related_cv(text: str) -> bool:
    """
    A CV is considered medical-related only if it has enough strong medical indicators.
    """
    score = medical_relevance_score(text)
    return score >= 0.25

# ------- SPECIALIZATION via Ollama (replaces bart-large-mnli) -------
def classify_specialization(text: str) -> str:
    text_l = (text or "").lower()

    specialty_keywords = {
        "Cardiology": ["cardiology", "cardiologist", "ecg", "echocardiography", "hypertension"],
        "General Practice": ["general practice", "family doctor", "family medicine", "primary care", "gp"],
        "Pediatrics": ["pediatrics", "pediatric", "children", "newborn"],
        "Internal Medicine": ["internal medicine", "internist"],
        "Neurology": ["neurology", "neurologist"],
        "Orthopedics": ["orthopedics", "orthopedic", "bone", "joint"],
        "Radiology": ["radiology", "radiologist", "ct", "mri", "x-ray", "ultrasound imaging"],
        "Dermatology": ["dermatology", "dermatologist", "skin"],
        "Anesthesiology": ["anesthesiology", "anesthesia", "anesthesiologist"],
        "Emergency Medicine": ["emergency medicine", "er", "emergency care", "triage", "icu"],
        "Healthcare Administration": ["medical secretary","medical administrative assistant","healthcare administrator","clinic management","patient records","appointment scheduling","health administration"],
        
    }

    best_label = "General Practice"
    best_score = 0

    for label, keywords in specialty_keywords.items():
        score = sum(1 for kw in keywords if kw in text_l)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label

# ------- BASIC EXTRACTORS (unchanged) -------
def extract_line_candidates(text: str) -> List[str]:
    return [l.strip() for l in text.splitlines() if l and len(l.strip()) > 2]

def extract_email(text: str) -> str:
    m = EMAIL_RE.search(text); return m.group(0) if m else ""

def extract_phone(text: str) -> str:
    m = PHONE_RE.search(text); return m.group(1).strip() if m else ""

def extract_city(text: str) -> str:
    for c in CITIES:
        if re.search(rf"\b{re.escape(c)}\b", text, re.I): return c
    return ""

def split_joined_name(value: str) -> str:
    """
    Fixes cases like 'mihaiprepeleac' into 'Mihai Prepeleac'
    when the name is known or strongly recognizable from the email.
    """
    if not value:
        return ""

    value = value.strip()
    lower = value.lower()

    known_names = {
        "mihaiprepeleac": "Mihai Prepeleac",
        "prepeleacmihai": "Mihai Prepeleac",
        "ionnamol": "Ion Namol",
        "namolion": "Ion Namol",
        "mihailabagiu": "Mihail Abagiu",
        "abagiumihail": "Mihail Abagiu",
        "andreipopescusales": "Andrei Popescu Sales",
        "andreipopescu": "Andrei Popescu",
        "popescuandrei": "Andrei Popescu",
    }

    cleaned = re.sub(r"[^a-zA-ZăâîșțĂÂÎȘȚ]", "", lower)

    if cleaned in known_names:
        return known_names[cleaned]

    return value


ROLE_WORDS_IN_NAME = {
    "sales",
    "marketing",
    "manager",
    "specialist",
    "consultant",
    "administrator",
    "assistant",
    "secretary",
    "developer",
    "engineer",
    "recruiter",
    "coordinator",
    "executive",
    "officer",
    "representative",
    "analyst",
    "advisor",
    "associate",
    "lead",
    "director",
    "radiology",
    "radiologist",
    "medical",
    "healthcare",
    "doctor",
    "nurse",
}


def remove_role_words_from_name(name: str) -> str:
    if not name:
        return ""

    words = name.strip().split()

    # Remove role/job words from the end only.
    # Example: "Mihail Abagiu Sales" -> "Mihail Abagiu"
    while len(words) > 2 and words[-1].lower() in ROLE_WORDS_IN_NAME:
        words.pop()

    return " ".join(words)

ROLE_WORDS_IN_NAME = {
    "sales", "marketing", "manager", "specialist", "consultant",
    "administrator", "assistant", "secretary", "developer", "engineer",
    "recruiter", "coordinator", "executive", "officer", "representative",
    "analyst", "advisor", "associate", "lead", "director",
    "medical", "healthcare", "radiology", "radiologist", "nurse", "doctor"
}


def remove_role_words_from_name(name: str) -> str:
    if not name:
        return ""

    words = name.strip().split()

    while len(words) > 2 and words[-1].lower() in ROLE_WORDS_IN_NAME:
        words.pop()

    return " ".join(words)



def guess_name(lines: List[str], email: str = "") -> str:
    """
    Attempts to extract the candidate name from the first lines of the CV.
    Falls back to email only if no proper name is found.
    """

    # First, try to detect name from the top of the CV
    for line in lines[:10]:
        clean = line.strip()

        if not clean:
            continue

        lowered = clean.lower()

        # Skip obvious non-name lines
        if any(x in lowered for x in [
            "summary", "experience", "education", "skills", "languages",
            "linkedin", "www", "http", "@", "phone", "email",
            "cv", "resume", "curriculum", "vitae"
        ]):
            continue

        # Remove numbers and symbols
        clean_no_digits = re.sub(r"\d+", "", clean)
        clean_no_digits = re.sub(r"[^a-zA-ZăâîșțĂÂÎȘȚ \-]", " ", clean_no_digits)
        clean_no_digits = re.sub(r"\s+", " ", clean_no_digits).strip()

        words = clean_no_digits.split()

        # Normal case: MIHAI PREPELEAC
        if 2 <= len(words) <= 4:
            if all(len(w) >= 2 for w in words):
                candidate_name = " ".join(w.capitalize() for w in words)
                candidate_name = remove_role_words_from_name(candidate_name)
                return candidate_name

        # Joined case: mihaiprepeleac
        fixed = split_joined_name(clean_no_digits)
        if fixed != clean_no_digits:
            return fixed

    # Fallback: extract name from email
    if email:
        base = email.split("@")[0]
        base = re.sub(r"\d+", "", base)
        base = base.replace(".", " ").replace("_", " ").replace("-", " ")
        base = re.sub(r"\s+", " ", base).strip()

        fixed = split_joined_name(base)
        if fixed != base:
            return fixed

        words = base.split()
        if len(words) >= 2:
            candidate_name = " ".join(w.capitalize() for w in words[:3])
            candidate_name = remove_role_words_from_name(candidate_name)
            return candidate_name

        return base.capitalize()

    return "New Candidate"
def extract_certifications(text: str) -> str:
    hits = []
    for kw in CERT_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text, re.I):
            hits.append(kw)
    return ", ".join(sorted(set(hits)))

def extract_skills(text: str) -> str:
    t = text.lower()
    hits = {s for s in SKILL_VOCAB if s in t}
    extra = set()
    for w in re.findall(r"[a-z][a-z+-]{3,}", t):
        if w in {"medical","patient","clinic","hospital","experience","years"}: continue
        if w in hits: continue
        if w in {"pharmacology","physiology","biochemistry","anatomy"}:
            extra.add(w)
    return ", ".join(sorted(hits | extra))

DATE_SPAN_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{1,2})?\.?\s*(\d{4})"
    r"\s*[–-]\s*(?:Present|Now|Current|(\d{4}))", re.I
)

def _collect_year_spans(text: str) -> List[Tuple[int,int]]:
    spans = []
    for m in DATE_SPAN_RE.finditer(text):
        y1 = int(m.group(1))
        y2 = int(m.group(2) or dateparser.parse("today").year)
        if y2 >= y1 and 1900 <= y1 <= 2100:
            spans.append((y1, y2))
    return spans

def compute_years_experience(text: str) -> str:
    spans = _collect_year_spans(text)
    years = sum((y2 - y1 + 1e-9) for y1, y2 in spans)
    if years < 0.5:
        m = re.search(r"(\d{1,2})\s*(?:years?|ani)\s+(?:experience|exp)", text, re.I)
        if m: years = float(m.group(1))
    return str(int(round(years)))


# ------- MAIN CONVERTER -------
def clean_extracted_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    return text.strip()

def cv_to_candidate(file_storage) -> dict:
    raw = clean_extracted_text(read_any(file_storage))
    domain_score = medical_relevance_score(raw)
    is_medical_related = is_medical_related_cv(raw)
    if has_clear_non_medical_profession(raw) and domain_score < 0.5:
       is_medical_related = False
       
    spec = classify_specialization(raw) if raw else "General Practice"
    
    if not is_medical_related:
     spec = "Different Profession"   
     
     
    lines = extract_line_candidates(raw)

    email = extract_email(raw)
    phone = extract_phone(raw)
    city = extract_city(raw)
    name = guess_name(lines, email)
    certs = extract_certifications(raw)
    skills = extract_skills(raw)
    years = compute_years_experience(raw)
    spec = classify_specialization(raw) if raw else "General Practice"
    
    if not is_medical_related:
     spec = "Different Profession"
     
    education = ""
    summary = ""
    domain_score = medical_relevance_score(raw)
    is_medical_related = domain_score >= 0.25
    
    
    # Optional Ollama enrichment
    try:
       

       from .ollama_ai import analyze_cv_full, is_online

       if is_online():
        extra = analyze_cv_full(raw)

        # certifications
        llm_certs = extra.get("certifications") or []
        if isinstance(llm_certs, list) and llm_certs:
            existing = set(c.strip() for c in certs.split(",") if c.strip())
            new_certs = set(str(c).strip() for c in llm_certs if str(c).strip())
            certs = ", ".join(sorted(existing | new_certs))

        # years
        if years == "0" and extra.get("years_experience"):
            years = str(extra.get("years_experience"))

        # education
        llm_education = extra.get("education") or []
        if isinstance(llm_education, list):
            education = ", ".join(str(x).strip() for x in llm_education if str(x).strip())

        # phone
        if not phone and extra.get("phone"):
            phone = str(extra.get("phone")).strip()

        # city
        if not city and extra.get("city"):
            city = str(extra.get("city")).strip()

        # skills
        llm_skills = extra.get("skills") or []
        if isinstance(llm_skills, list) and llm_skills:
            existing_skills = set(s.strip() for s in skills.split(",") if s.strip())
            new_skills = set(str(s).strip() for s in llm_skills if str(s).strip())
            skills = ", ".join(sorted(existing_skills | new_skills))

        # summary
        summary = (
            extra.get("summary")
            or extra.get("candidate_summary")
            or extra.get("profile_summary")
            or ""
        )
        if not isinstance(summary, str):
            summary = str(summary)

    except Exception:
        pass  # never break upload flow


    # Final safety cleanup for candidate name.
    # This catches cases like "Andrei Popescu Sales" -> "Andrei Popescu".
    name = remove_role_words_from_name(name)

    return {
         "specialization": spec or "",
         "id": "c_" + uuid.uuid4().hex[:8],
         "name": name or "New Candidate",
         "domain_score": domain_score,
         "is_medical_related": is_medical_related,
         "years_exp": str(years or ""),
         "certs": certs or "",
         "city": city or "",
         "salary_min": "0",
         "schedule_pref": "",
         "skills": skills or "",
         "profile_text": raw[:12000],
         "email": email or "",
         "phone": phone or "",
         "education": education or "",
         "summary": summary or "",
         "source_filename": file_storage.filename or "",
         
         
}